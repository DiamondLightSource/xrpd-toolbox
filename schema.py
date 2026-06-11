import sgqlc.types
import sgqlc.types.datetime
import sgqlc.types.relay

schema = sgqlc.types.Schema()


# Unexport Node/PageInfo, let schema re-declare them
schema -= sgqlc.types.relay.Node
schema -= sgqlc.types.relay.PageInfo


########################################################################
# Scalars and Enumerations
########################################################################
Boolean = sgqlc.types.Boolean


class Creator(sgqlc.types.Scalar):
    __schema__ = schema


DateTime = sgqlc.types.datetime.DateTime

Float = sgqlc.types.Float

ID = sgqlc.types.ID

Int = sgqlc.types.Int


class JSON(sgqlc.types.Scalar):
    __schema__ = schema


class JSONObject(sgqlc.types.Scalar):
    __schema__ = schema


class ScienceGroup(sgqlc.types.Enum):
    __schema__ = schema
    __choices__ = (
        "BIO_CRYO_IMAGING",
        "CONDENSED_MATTER",
        "CRYSTALLOGRAPHY",
        "EXAMPLES",
        "IMAGING",
        "MAGNETIC_MATERIALS",
        "MX",
        "SPECTROSCOPY",
        "SURFACES",
    )


String = sgqlc.types.String


class TaskStatus(sgqlc.types.Enum):
    __schema__ = schema
    __choices__ = (
        "ERROR",
        "FAILED",
        "OMITTED",
        "PENDING",
        "RUNNING",
        "SKIPPED",
        "SUCCEEDED",
    )


class Template(sgqlc.types.Scalar):
    __schema__ = schema


class Url(sgqlc.types.Scalar):
    __schema__ = schema


########################################################################
# Input Objects
########################################################################
class VisitInput(sgqlc.types.Input):
    __schema__ = schema
    __field_names__ = ("proposal_code", "proposal_number", "number")
    proposal_code = sgqlc.types.Field(
        sgqlc.types.non_null(String), graphql_name="proposalCode"
    )
    proposal_number = sgqlc.types.Field(
        sgqlc.types.non_null(Int), graphql_name="proposalNumber"
    )
    number = sgqlc.types.Field(sgqlc.types.non_null(Int), graphql_name="number")


class WorkflowFilter(sgqlc.types.Input):
    __schema__ = schema
    __field_names__ = ("workflow_status_filter", "creator", "template")
    workflow_status_filter = sgqlc.types.Field(
        "WorkflowStatusFilter", graphql_name="workflowStatusFilter"
    )
    creator = sgqlc.types.Field(Creator, graphql_name="creator")
    template = sgqlc.types.Field(Template, graphql_name="template")


class WorkflowStatusFilter(sgqlc.types.Input):
    __schema__ = schema
    __field_names__ = ("pending", "running", "succeeded", "failed", "error")
    pending = sgqlc.types.Field(sgqlc.types.non_null(Boolean), graphql_name="pending")
    running = sgqlc.types.Field(sgqlc.types.non_null(Boolean), graphql_name="running")
    succeeded = sgqlc.types.Field(
        sgqlc.types.non_null(Boolean), graphql_name="succeeded"
    )
    failed = sgqlc.types.Field(sgqlc.types.non_null(Boolean), graphql_name="failed")
    error = sgqlc.types.Field(sgqlc.types.non_null(Boolean), graphql_name="error")


class WorkflowTemplatesFilter(sgqlc.types.Input):
    __schema__ = schema
    __field_names__ = ("science_group",)
    science_group = sgqlc.types.Field(
        sgqlc.types.list_of(sgqlc.types.non_null(ScienceGroup)),
        graphql_name="scienceGroup",
    )


########################################################################
# Output Objects and Interfaces
########################################################################
class Artifact(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("name", "url", "mime_type")
    name = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="name")
    url = sgqlc.types.Field(sgqlc.types.non_null(Url), graphql_name="url")
    mime_type = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="mimeType")


class LogEntry(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("content", "pod_name")
    content = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="content")
    pod_name = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="podName")


class Mutation(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("submit_workflow_template",)
    submit_workflow_template = sgqlc.types.Field(
        sgqlc.types.non_null("Workflow"),
        graphql_name="submitWorkflowTemplate",
        args=sgqlc.types.ArgDict(
            (
                (
                    "name",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(String), graphql_name="name", default=None
                    ),
                ),
                (
                    "visit",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(VisitInput),
                        graphql_name="visit",
                        default=None,
                    ),
                ),
                (
                    "parameters",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(JSON),
                        graphql_name="parameters",
                        default=None,
                    ),
                ),
            )
        ),
    )


class PageInfo(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = (
        "has_previous_page",
        "has_next_page",
        "start_cursor",
        "end_cursor",
    )
    has_previous_page = sgqlc.types.Field(
        sgqlc.types.non_null(Boolean), graphql_name="hasPreviousPage"
    )
    has_next_page = sgqlc.types.Field(
        sgqlc.types.non_null(Boolean), graphql_name="hasNextPage"
    )
    start_cursor = sgqlc.types.Field(String, graphql_name="startCursor")
    end_cursor = sgqlc.types.Field(String, graphql_name="endCursor")


class Query(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = (
        "node",
        "workflow",
        "workflows",
        "workflow_template",
        "workflow_templates",
        "_service",
    )
    node = sgqlc.types.Field(
        "NodeValue",
        graphql_name="node",
        args=sgqlc.types.ArgDict(
            (
                (
                    "id",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(ID), graphql_name="id", default=None
                    ),
                ),
            )
        ),
    )
    workflow = sgqlc.types.Field(
        sgqlc.types.non_null("Workflow"),
        graphql_name="workflow",
        args=sgqlc.types.ArgDict(
            (
                (
                    "visit",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(VisitInput),
                        graphql_name="visit",
                        default=None,
                    ),
                ),
                (
                    "name",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(String), graphql_name="name", default=None
                    ),
                ),
            )
        ),
    )
    workflows = sgqlc.types.Field(
        sgqlc.types.non_null("WorkflowConnection"),
        graphql_name="workflows",
        args=sgqlc.types.ArgDict(
            (
                (
                    "visit",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(VisitInput),
                        graphql_name="visit",
                        default=None,
                    ),
                ),
                (
                    "cursor",
                    sgqlc.types.Arg(String, graphql_name="cursor", default=None),
                ),
                ("limit", sgqlc.types.Arg(Int, graphql_name="limit", default=None)),
                (
                    "filter",
                    sgqlc.types.Arg(
                        WorkflowFilter, graphql_name="filter", default=None
                    ),
                ),
            )
        ),
    )
    workflow_template = sgqlc.types.Field(
        sgqlc.types.non_null("WorkflowTemplate"),
        graphql_name="workflowTemplate",
        args=sgqlc.types.ArgDict(
            (
                (
                    "name",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(String), graphql_name="name", default=None
                    ),
                ),
            )
        ),
    )
    workflow_templates = sgqlc.types.Field(
        sgqlc.types.non_null("WorkflowTemplateConnection"),
        graphql_name="workflowTemplates",
        args=sgqlc.types.ArgDict(
            (
                (
                    "cursor",
                    sgqlc.types.Arg(String, graphql_name="cursor", default=None),
                ),
                ("limit", sgqlc.types.Arg(Int, graphql_name="limit", default=None)),
                (
                    "filter",
                    sgqlc.types.Arg(
                        WorkflowTemplatesFilter, graphql_name="filter", default=None
                    ),
                ),
            )
        ),
    )
    _service = sgqlc.types.Field(
        sgqlc.types.non_null("_Service"), graphql_name="_service"
    )


class Subscription(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("logs", "workflow")
    logs = sgqlc.types.Field(
        sgqlc.types.non_null(LogEntry),
        graphql_name="logs",
        args=sgqlc.types.ArgDict(
            (
                (
                    "visit",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(VisitInput),
                        graphql_name="visit",
                        default=None,
                    ),
                ),
                (
                    "workflow_name",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(String),
                        graphql_name="workflowName",
                        default=None,
                    ),
                ),
                (
                    "task_id",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(String),
                        graphql_name="taskId",
                        default=None,
                    ),
                ),
            )
        ),
    )
    workflow = sgqlc.types.Field(
        sgqlc.types.non_null("Workflow"),
        graphql_name="workflow",
        args=sgqlc.types.ArgDict(
            (
                (
                    "visit",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(VisitInput),
                        graphql_name="visit",
                        default=None,
                    ),
                ),
                (
                    "name",
                    sgqlc.types.Arg(
                        sgqlc.types.non_null(String), graphql_name="name", default=None
                    ),
                ),
            )
        ),
    )


class Task(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = (
        "id",
        "name",
        "status",
        "depends",
        "dependencies",
        "artifacts",
        "step_type",
        "start_time",
        "end_time",
        "message",
    )
    id = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="id")
    name = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="name")
    status = sgqlc.types.Field(sgqlc.types.non_null(TaskStatus), graphql_name="status")
    depends = sgqlc.types.Field(
        sgqlc.types.non_null(sgqlc.types.list_of(sgqlc.types.non_null(String))),
        graphql_name="depends",
    )
    dependencies = sgqlc.types.Field(
        sgqlc.types.non_null(sgqlc.types.list_of(sgqlc.types.non_null(String))),
        graphql_name="dependencies",
    )
    artifacts = sgqlc.types.Field(
        sgqlc.types.non_null(sgqlc.types.list_of(sgqlc.types.non_null(Artifact))),
        graphql_name="artifacts",
    )
    step_type = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="stepType")
    start_time = sgqlc.types.Field(DateTime, graphql_name="startTime")
    end_time = sgqlc.types.Field(DateTime, graphql_name="endTime")
    message = sgqlc.types.Field(String, graphql_name="message")


class TemplateSource(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("repository_url", "path", "target_revision")
    repository_url = sgqlc.types.Field(
        sgqlc.types.non_null(String), graphql_name="repositoryUrl"
    )
    path = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="path")
    target_revision = sgqlc.types.Field(
        sgqlc.types.non_null(String), graphql_name="targetRevision"
    )


class Visit(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("proposal_code", "proposal_number", "number")
    proposal_code = sgqlc.types.Field(
        sgqlc.types.non_null(String), graphql_name="proposalCode"
    )
    proposal_number = sgqlc.types.Field(
        sgqlc.types.non_null(Int), graphql_name="proposalNumber"
    )
    number = sgqlc.types.Field(sgqlc.types.non_null(Int), graphql_name="number")


class Workflow(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = (
        "id",
        "name",
        "visit",
        "status",
        "parameters",
        "template_ref",
        "creator",
    )
    id = sgqlc.types.Field(sgqlc.types.non_null(ID), graphql_name="id")
    name = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="name")
    visit = sgqlc.types.Field(sgqlc.types.non_null(Visit), graphql_name="visit")
    status = sgqlc.types.Field("WorkflowStatus", graphql_name="status")
    parameters = sgqlc.types.Field(JSONObject, graphql_name="parameters")
    template_ref = sgqlc.types.Field(String, graphql_name="templateRef")
    creator = sgqlc.types.Field(
        sgqlc.types.non_null("WorkflowCreator"), graphql_name="creator"
    )


class WorkflowConnection(sgqlc.types.relay.Connection):
    __schema__ = schema
    __field_names__ = ("page_info", "edges", "nodes")
    page_info = sgqlc.types.Field(
        sgqlc.types.non_null(PageInfo), graphql_name="pageInfo"
    )
    edges = sgqlc.types.Field(
        sgqlc.types.non_null(sgqlc.types.list_of(sgqlc.types.non_null("WorkflowEdge"))),
        graphql_name="edges",
    )
    nodes = sgqlc.types.Field(
        sgqlc.types.non_null(sgqlc.types.list_of(sgqlc.types.non_null(Workflow))),
        graphql_name="nodes",
    )


class WorkflowCreator(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("creator_id",)
    creator_id = sgqlc.types.Field(
        sgqlc.types.non_null(String), graphql_name="creatorId"
    )


class WorkflowEdge(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("node", "cursor")
    node = sgqlc.types.Field(sgqlc.types.non_null(Workflow), graphql_name="node")
    cursor = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="cursor")


class WorkflowErroredStatus(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("start_time", "end_time", "message", "tasks")
    start_time = sgqlc.types.Field(
        sgqlc.types.non_null(DateTime), graphql_name="startTime"
    )
    end_time = sgqlc.types.Field(sgqlc.types.non_null(DateTime), graphql_name="endTime")
    message = sgqlc.types.Field(String, graphql_name="message")
    tasks = sgqlc.types.Field(
        sgqlc.types.non_null(sgqlc.types.list_of(sgqlc.types.non_null(Task))),
        graphql_name="tasks",
    )


class WorkflowFailedStatus(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("start_time", "end_time", "message", "tasks")
    start_time = sgqlc.types.Field(
        sgqlc.types.non_null(DateTime), graphql_name="startTime"
    )
    end_time = sgqlc.types.Field(sgqlc.types.non_null(DateTime), graphql_name="endTime")
    message = sgqlc.types.Field(String, graphql_name="message")
    tasks = sgqlc.types.Field(
        sgqlc.types.non_null(sgqlc.types.list_of(sgqlc.types.non_null(Task))),
        graphql_name="tasks",
    )


class WorkflowPendingStatus(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("message",)
    message = sgqlc.types.Field(String, graphql_name="message")


class WorkflowRunningStatus(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("start_time", "message", "tasks")
    start_time = sgqlc.types.Field(
        sgqlc.types.non_null(DateTime), graphql_name="startTime"
    )
    message = sgqlc.types.Field(String, graphql_name="message")
    tasks = sgqlc.types.Field(
        sgqlc.types.non_null(sgqlc.types.list_of(sgqlc.types.non_null(Task))),
        graphql_name="tasks",
    )


class WorkflowSucceededStatus(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("start_time", "end_time", "message", "tasks")
    start_time = sgqlc.types.Field(
        sgqlc.types.non_null(DateTime), graphql_name="startTime"
    )
    end_time = sgqlc.types.Field(sgqlc.types.non_null(DateTime), graphql_name="endTime")
    message = sgqlc.types.Field(String, graphql_name="message")
    tasks = sgqlc.types.Field(
        sgqlc.types.non_null(sgqlc.types.list_of(sgqlc.types.non_null(Task))),
        graphql_name="tasks",
    )


class WorkflowTemplate(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = (
        "name",
        "maintainer",
        "title",
        "description",
        "repository",
        "arguments",
        "ui_schema",
        "template_source",
    )
    name = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="name")
    maintainer = sgqlc.types.Field(
        sgqlc.types.non_null(String), graphql_name="maintainer"
    )
    title = sgqlc.types.Field(String, graphql_name="title")
    description = sgqlc.types.Field(String, graphql_name="description")
    repository = sgqlc.types.Field(String, graphql_name="repository")
    arguments = sgqlc.types.Field(sgqlc.types.non_null(JSON), graphql_name="arguments")
    ui_schema = sgqlc.types.Field(JSON, graphql_name="uiSchema")
    template_source = sgqlc.types.Field(TemplateSource, graphql_name="templateSource")


class WorkflowTemplateConnection(sgqlc.types.relay.Connection):
    __schema__ = schema
    __field_names__ = ("page_info", "edges", "nodes")
    page_info = sgqlc.types.Field(
        sgqlc.types.non_null(PageInfo), graphql_name="pageInfo"
    )
    edges = sgqlc.types.Field(
        sgqlc.types.non_null(
            sgqlc.types.list_of(sgqlc.types.non_null("WorkflowTemplateEdge"))
        ),
        graphql_name="edges",
    )
    nodes = sgqlc.types.Field(
        sgqlc.types.non_null(
            sgqlc.types.list_of(sgqlc.types.non_null(WorkflowTemplate))
        ),
        graphql_name="nodes",
    )


class WorkflowTemplateEdge(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("node", "cursor")
    node = sgqlc.types.Field(
        sgqlc.types.non_null(WorkflowTemplate), graphql_name="node"
    )
    cursor = sgqlc.types.Field(sgqlc.types.non_null(String), graphql_name="cursor")


class _Service(sgqlc.types.Type):
    __schema__ = schema
    __field_names__ = ("sdl",)
    sdl = sgqlc.types.Field(String, graphql_name="sdl")


########################################################################
# Unions
########################################################################
class NodeValue(sgqlc.types.Union):
    __schema__ = schema
    __types__ = (Workflow,)


class WorkflowStatus(sgqlc.types.Union):
    __schema__ = schema
    __types__ = (
        WorkflowPendingStatus,
        WorkflowRunningStatus,
        WorkflowSucceededStatus,
        WorkflowFailedStatus,
        WorkflowErroredStatus,
    )


########################################################################
# Schema Entry Points
########################################################################
schema.query_type = Query
schema.mutation_type = Mutation
schema.subscription_type = Subscription
