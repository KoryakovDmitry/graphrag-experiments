# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'PipelineReportingConfig', 'PipelineFileReportingConfig' and 'PipelineConsoleReportingConfig' models."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel
from pydantic import Field as pydantic_Field

from graphrag.config.enums import ReportingType

T = TypeVar("T")


class PipelineReportingConfig(BaseModel, Generic[T]):
    """Represent the reporting configuration for the pipeline."""

    type: T


class PipelineFileReportingConfig(PipelineReportingConfig[Literal[ReportingType.file]]):
    """Represent the file reporting configuration for the pipeline."""

    type: Literal[ReportingType.file] = ReportingType.file
    """The type of reporting."""

    base_dir: str | None = pydantic_Field(
        description="The base directory for the reporting.", default=None
    )
    """The base directory for the reporting."""


class PipelineConsoleReportingConfig(
    PipelineReportingConfig[Literal[ReportingType.console]]
):
    """Represent the console reporting configuration for the pipeline."""

    type: Literal[ReportingType.console] = ReportingType.console
    """The type of reporting."""


class PipelineBlobReportingConfig(PipelineReportingConfig[Literal[ReportingType.blob]]):
    """Represents the blob reporting configuration for the pipeline."""

    type: Literal[ReportingType.blob] = ReportingType.blob
    """The type of reporting."""

    connection_string: str | None = pydantic_Field(
        description="The blob reporting connection string for the reporting.",
        default=None,
    )
    """The blob reporting connection string for the reporting."""

    container_name: str = pydantic_Field(
        description="The container name for reporting", default=None
    )
    """The container name for reporting"""

    storage_account_blob_url: str | None = pydantic_Field(
        description="The storage account blob url for reporting", default=None
    )
    """The storage account blob url for reporting"""

    base_dir: str | None = pydantic_Field(
        description="The base directory for the reporting.", default=None
    )
    """The base directory for the reporting."""


class PipelineS3ReportingConfig(PipelineReportingConfig[Literal[ReportingType.s3]]):
    """Represents the S3 reporting configuration for the pipeline."""

    type: Literal[ReportingType.s3] = ReportingType.s3
    """The type of reporting."""

    base_dir: str | None = pydantic_Field(
        description="The base directory for the reporting.", default=None
    )
    """The base directory for the reporting."""

    bucket_name: str = pydantic_Field(
        description="The S3 bucket name for reporting", default=None
    )
    """The S3 bucket name for reporting"""

    region_name: str | None = pydantic_Field(
        description="The region name for the S3 bucket", default=None
    )
    """The region name for the S3 bucket"""

    object_name: str | None = pydantic_Field(
        description="Log file name in S3 bucket", default=None
    )
    """Log file name in S3 bucket"""


PipelineReportingConfigTypes = (
        PipelineFileReportingConfig
        | PipelineConsoleReportingConfig
        | PipelineBlobReportingConfig
        | PipelineS3ReportingConfig
)
