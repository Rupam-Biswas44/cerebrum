"""
MinIO Object Storage Module

Handles file uploads, downloads, and presigned URLs for MinIO (S3 compatible).
"""

from io import BytesIO

import structlog
from minio import Minio
from minio.error import S3Error

from cerebrum.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Initialize MinIO client
try:
    minio_client = Minio(
        endpoint=f"{settings.MINIO_HOST}:{settings.MINIO_PORT}",
        access_key=settings.MINIO_ROOT_USER,
        secret_key=settings.MINIO_ROOT_PASSWORD,
        secure=False,  # Set True for production with TLS
    )
except Exception as e:
    logger.error("minio.init.failed", error=str(e))
    minio_client = None


def ensure_bucket_exists(bucket_name: str) -> None:
    """Ensure the target bucket exists, creating it if necessary."""
    if not minio_client:
        return
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logger.info("minio.bucket.created", bucket_name=bucket_name)
    except S3Error as e:
        logger.error("minio.bucket.check.failed", bucket_name=bucket_name, error=str(e))


def upload_file_stream(
    bucket_name: str, object_name: str, data: BytesIO, size: int, content_type: str
) -> str:
    """
    Upload a file stream to MinIO.

    Args:
        bucket_name: Target bucket.
        object_name: The destination path/name in the bucket.
        data: The BytesIO stream containing file data.
        size: Total size of the file in bytes.
        content_type: MIME type.

    Returns:
        The object name/path on success.
    """
    if not minio_client:
        raise RuntimeError("MinIO client not initialized")

    ensure_bucket_exists(bucket_name)

    try:
        minio_client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=data,
            length=size,
            content_type=content_type,
        )
        logger.info(
            "minio.upload.success", bucket_name=bucket_name, object_name=object_name, size=size
        )
        return f"s3://{bucket_name}/{object_name}"
    except S3Error as e:
        logger.error(
            "minio.upload.failed", bucket_name=bucket_name, object_name=object_name, error=str(e)
        )
        raise RuntimeError(f"Failed to upload to storage: {e}") from e


def get_presigned_url(bucket_name: str, object_name: str, expires_seconds: int = 3600) -> str:
    """Generate a presigned URL to securely download an object."""
    from datetime import timedelta

    if not minio_client:
        raise RuntimeError("MinIO client not initialized")

    try:
        url: str = minio_client.get_presigned_url(
            "GET",
            bucket_name,
            object_name,
            expires=timedelta(seconds=expires_seconds),
        )
        return url
    except S3Error as e:
        logger.error(
            "minio.presigned_url.failed",
            bucket_name=bucket_name,
            object_name=object_name,
            error=str(e),
        )
        raise RuntimeError(f"Failed to generate download URL: {e}") from e
