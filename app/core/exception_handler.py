from functools import wraps

from fastapi import HTTPException

from src.internal.common.exception import ExceptionInternalError
from src.core.logger import logger


def exception_handler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(e)
            raise ExceptionInternalError from e

    return wrapper
