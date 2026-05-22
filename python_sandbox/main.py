import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

try:
    import resource  # POSIX only — container is Linux, so always available here
except ImportError:
    resource = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

SANDBOX_MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
SANDBOX_CPU_TIME_BUFFER_SECONDS = 5

CONFIG_PATH = Path("config.yaml")
ENV_FILE_PATH = Path(".env")

CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
load_dotenv(ENV_FILE_PATH)

app = FastAPI(
    title="Python Sandbox API",
    description="API to execute Python code in a sandboxed environment",
)


class CodePayload(BaseModel):
    code: str = Field(..., description="Python code to execute.")
    timeout_seconds: int = Field(
        60, gt=0, le=600, description="Execution timeout in seconds (1-600)."
    )


class ExecutionResult(BaseModel):
    success: bool
    stdout: str
    stderr: str
    returncode: int


def _build_preexec(timeout_seconds: int):
    """Return a preexec_fn that caps memory + CPU before the child execs.

    LLM-generated code can attempt to allocate gigabytes or spin in a tight
    loop. Wall-clock timeout alone leaves the container under load for the
    full timeout; RLIMIT_AS + RLIMIT_CPU let the kernel kill misbehaving
    code immediately. POSIX-only; on non-POSIX the import guard above
    leaves ``resource`` as None and we skip the limits.
    """
    if resource is None:
        return None

    cpu_seconds = timeout_seconds + SANDBOX_CPU_TIME_BUFFER_SECONDS

    def _apply_limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (SANDBOX_MEMORY_LIMIT_BYTES, SANDBOX_MEMORY_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

    return _apply_limits


def _run_code(code: str, timeout_seconds: int) -> ExecutionResult:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = Path(temp_file.name)

    logger.info(
        "sandbox: starting subprocess (timeout=%ss, code_bytes=%d)",
        timeout_seconds,
        len(code),
    )
    try:
        result = subprocess.run(
            [sys.executable, str(temp_file_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            preexec_fn=_build_preexec(timeout_seconds),
        )
        logger.info(
            "sandbox: subprocess finished (returncode=%s, stdout_bytes=%d, stderr_bytes=%d)",
            result.returncode,
            len(result.stdout or ""),
            len(result.stderr or ""),
        )
        return ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired:
        logger.warning("sandbox: subprocess timed out after %ss", timeout_seconds)
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="Execution timed out.",
            returncode=124,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        logger.exception("sandbox: environment error")
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Execution environment error: {exc}",
            returncode=1,
        )
    finally:
        try:
            temp_file_path.unlink()
        except OSError as exc:
            logger.warning("sandbox: could not remove temp file %s: %s", temp_file_path, exc)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to the Python Sandbox API!"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/execute", response_model=ExecutionResult)
async def execute_code(payload: CodePayload) -> ExecutionResult:
    return await run_in_threadpool(_run_code, payload.code, payload.timeout_seconds)
