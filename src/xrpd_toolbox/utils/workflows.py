import asyncio

from python_workflow_submitter.submit_workflow import submit_workflow

if __name__ == "__main__":
    asyncio.run(
        submit_workflow(
            "example-template",
            {"png": True, "jpg": False, "jpeg": True, "tif": True, "tiff": False},
            visit="cm44163-3",
        )
    )
