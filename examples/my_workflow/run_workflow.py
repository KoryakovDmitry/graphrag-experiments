from dotenv import load_dotenv

load_dotenv()

import os
import asyncio
import time
from datashaper import AsyncType
from graphrag.config import (
    GraphRagConfig,
    ReportingConfig,
    ReportingType,
    StorageConfig,
    StorageType,
    CacheConfig,
    CacheType,
    InputConfig,
    InputType,
    InputFileType,
    EmbedGraphConfig,
    TextEmbeddingConfig,
    LLMParameters,
    LLMType,
    ChunkingConfig,
    SnapshotsConfig,
    EntityExtractionConfig,
    SummarizeDescriptionsConfig,
    ClaimExtractionConfig,
    CommunityReportsConfig,
    ClusterGraphConfig,
    LocalSearchConfig,
    GlobalSearchConfig,
    UmapConfig,
    ParallelizationParameters,
)
import graphrag.config.defaults as defs
from examples.my_workflow.utils import (
    # _get_progress_reporter,
    _enable_logging,
    _get_progress_reporter,
)
from graphrag.index.emit import TableEmitterType
from graphrag.index import (
    create_pipeline_config,
    run_pipeline_with_config,
    PipelineConfig,
)


async def main():
    if "OPENAI_API_KEY" not in os.environ:
        msg = "Please set OPENAI_API_KEY environment variable to run this example"
        raise Exception(msg)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    reporter_type = "rich"
    emit = [
        TableEmitterType.Parquet,
    ]  # parquet, json, csv
    verbose = True
    memory_profile = False
    resume = False
    experiment_name = "car-v0"

    settings = GraphRagConfig(
        root_dir=root_dir,
        reporting=ReportingConfig(
            type=ReportingType.s3,
            # type=ReportingType.console,
            base_dir=f"output/{time.strftime('%Y%m%d-%H%M%S')}/reports",
            bucket_name="aiuc-dev-1",
            region_name="eu-north-1",
        ),
        storage=StorageConfig(
            type=StorageType.s3,
            base_dir=f"output/{time.strftime('%Y%m%d-%H%M%S')}/artifacts",
            bucket_name="aiuc-dev-1",
            region_name="eu-north-1",
        ),
        cache=CacheConfig(
            type=CacheType.file,
            base_dir=f"cache_{experiment_name}",
        ),
        # cache=CacheConfig(
        #     type=CacheType.s3,
        #     base_dir=f"cache_{experiment_name}",
        #     bucket_name="aiuc-dev-1",
        #     region_name="eu-north-1",
        # ),
        # input=InputConfig(
        #     type=InputType.file,
        #     file_type=InputFileType.text,
        #     base_dir="input",
        #     encoding="utf-8",
        #     file_pattern=".*\\.txt$",
        # ),
        input=InputConfig(
            type=InputType.file,
            file_type=InputFileType.csv,
            base_dir="input",
            encoding="utf-8",
            file_pattern=".*\\.csv$",
            source_column="url",
            text_column="content",
            # timestamp_column="date(yyyyMMddHHmmss)",
            # timestamp_format="%Y%m%d%H%M%S",
            title_column="title",
            document_attribute_columns=["id", "model_name", "make_name", "src"]
        ),
        embed_graph=EmbedGraphConfig(
            enabled=False,
        ),
        embeddings=TextEmbeddingConfig(
            llm=LLMParameters(
                api_key=os.getenv("OPENAI_API_KEY"),
                type=LLMType.OpenAIEmbedding,
                model="text-embedding-3-small",
            ),
            vector_store=None,
        ),
        chunks=ChunkingConfig(size=300, overlap=100),
        snapshots=SnapshotsConfig(
            graphml=True, raw_entities=True, top_level_nodes=True
        ),
        entity_extraction=EntityExtractionConfig(
            llm=LLMParameters(
                api_key=os.getenv("OPENAI_API_KEY"),
                type=LLMType.OpenAIChat,
                model="gpt-4o",
            ),
            prompt="prompts/entity_extraction.txt",
            # entity_types=["organization", "person", "geo", "event"],
            entity_types=[
                "Accidents",
                "Author",
                "Body",
                "Car",
                "Competitors",
                "Costs",
                "Drivetrain",
                "Engine",
                "Exterior",
                "Features",
                "Generation",
                "Interior",
                "Make",
                "Mileage",
                "Model",
                "Opinion",
                "Owners",
                "Package",
                "Performance",
                "Predecessor",
                "Segment",
                "Series",
                "Size",
                "Source",
                "Successor",
                "Transmission",
                "User",
                "Weight",
                "Year"
            ],
            max_gleanings=0,
        ),
        summarize_descriptions=SummarizeDescriptionsConfig(
            llm=LLMParameters(
                api_key=os.getenv("OPENAI_API_KEY"),
                type=LLMType.OpenAIChat,
                model="gpt-4o",
            ),
            prompt="prompts/summarize_descriptions.txt",
            max_length=500,
        ),
        community_reports=CommunityReportsConfig(
            llm=LLMParameters(
                api_key=os.getenv("OPENAI_API_KEY"),
                type=LLMType.OpenAIChat,
                model="gpt-4o",
            ),
            prompt="prompts/community_report.txt",
            max_length=2000,
            max_input_length=100_000,
        ),
        claim_extraction=ClaimExtractionConfig(
            llm=LLMParameters(
                api_key=os.getenv("OPENAI_API_KEY"),
                type=LLMType.OpenAIChat,
                model="gpt-4o",
            ),
            prompt="prompts/claim_extraction.txt",
            description=defs.CLAIM_DESCRIPTION,
            max_gleanings=0,
            enabled=True,
        ),
        cluster_graph=ClusterGraphConfig(max_cluster_size=10),
        umap=UmapConfig(enabled=False),
        local_search=LocalSearchConfig(),
        global_search=GlobalSearchConfig(),
        encoding_model="cl100k_base",
        skip_workflows=[],
        parallelization=ParallelizationParameters(stagger=0.3, num_threads=50),
        async_mode=AsyncType.Threaded,
    )

    pipeline_cfg: PipelineConfig = create_pipeline_config(settings=settings)

    progress_reporter = _get_progress_reporter(reporter_type=reporter_type)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    _enable_logging(root_dir, run_id, verbose)

    outputs = []
    # encountered_errors = False
    async for output in run_pipeline_with_config(
            pipeline_cfg,
            run_id=run_id,
            memory_profile=memory_profile,
            progress_reporter=progress_reporter,
            emit=emit,
            is_resume_run=resume,
    ):
        # if output.errors and len(output.errors) > 0:
        #     encountered_errors = True
        #     progress_reporter.error(output.workflow)
        # else:
        #     progress_reporter.success(output.workflow)

        # progress_reporter.info(str(output.result))
        outputs.append(output)

    # progress_reporter.stop()
    # if encountered_errors:
    #     progress_reporter.error(
    #         "Errors occurred during the pipeline run, see logs for more details."
    #     )
    # else:
    #     progress_reporter.success("All workflows completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
