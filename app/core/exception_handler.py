from functools import wraps

from fastapi import HTTPException

from app.utils.logger import setup_logging

logger = setup_logging("Exception")


def exception_handler(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(e)
            raise e

    return wrapper
