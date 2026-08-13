"""
Shared configuration for the NLP extraction pipeline.

Defines the 13-category skill taxonomy (derived from ESCO and O*NET, corpus-validated)
and the Tier 2 lexicons used by the lexicon-based methods. All four extraction methods
import from here so that the label space is identical across methods and identical to
the gold standard.

See annotation_guidelines_v2_0.docx for category definitions and boundary rules.
"""

# Category order is fixed: it defines the column order in every prediction matrix.
CATEGORIES = [
    "programming",
    "sql",
    "visualisation_bi",
    "reporting",
    "excel",
    "statistics",
    "machine_learning",
    "data_cleaning",
    "etl",
    "data_modelling",
    "cloud",
    "stakeholder_comm",
    "ethics_governance",
]

# Human-readable names, used as zero-shot hypotheses and in LLM prompts.
CATEGORY_LABELS = {
    "programming": "programming or scripting languages",
    "sql": "database querying with SQL",
    "visualisation_bi": "data visualisation and business intelligence tools",
    "reporting": "producing reports and management information",
    "excel": "spreadsheet software such as Excel",
    "statistics": "statistical analysis and forecasting",
    "machine_learning": "machine learning and predictive modelling",
    "data_cleaning": "data cleaning and data quality",
    "etl": "data engineering, ETL pipelines and data warehousing",
    "data_modelling": "data modelling and schema design",
    "cloud": "cloud computing platforms",
    "stakeholder_comm": "stakeholder communication and presenting findings",
    "ethics_governance": "data governance, privacy and GDPR compliance",
}

# Tier 2 lexicons. Seed lists from ESCO/O*NET, extended during corpus validation
# and annotation. Terms are matched case-insensitively as whole words.
LEXICONS = {
    "programming": [
        "python", "r programming", "scala", "java", "c#", "vba", "macros", "dax",
        "pyspark", "bash", "shell scripting", "powershell", "programming",
        "coding", "scripting", "programming languages",
    ],
    "sql": [
        "sql", "t-sql", "transact-sql", "pl/sql", "mysql", "postgresql", "postgres",
        "sql server", "oracle database", "mongodb", "nosql", "bigquery", "database",
        "databases", "relational database", "relational databases", "ms access",
        "microsoft access", "dynamodb", "elasticsearch", "hive", "cassandra", "db2",
    ],
    "visualisation_bi": [
        "power bi", "powerbi", "tableau", "looker", "looker studio", "qlik",
        "qlikview", "alteryx", "spotfire", "data visualisation", "data visualization",
        "visualisation", "visualization", "dashboard", "dashboards",
    ],
    "reporting": [
        "reporting", "reports", "report writing", "report production", "mi",
        "management information", "management reporting", "kpi reporting",
        "performance pack", "ssrs", "crystal reports", "sisense",
    ],
    "excel": [
        "excel", "microsoft excel", "ms excel", "google sheets", "spreadsheet",
        "spreadsheets", "pivot table", "pivot tables", "vlookup", "vlookups",
        "power query", "powerquery",
    ],
    "statistics": [
        "statistics", "statistical", "statistical analysis", "statistical modelling",
        "statistical modeling", "regression", "forecasting", "forecast", "forecasts",
        "time series", "hypothesis testing", "a/b testing", "a/b test",
        "econometrics", "econometric", "sas", "spss", "matlab", "minitab",
    ],
    "machine_learning": [
        "machine learning", "ml", "predictive modelling", "predictive modeling",
        "predictive analytics", "feature engineering", "deep learning", "nlp",
        "natural language processing", "llm", "tensorflow", "pytorch",
        "scikit-learn", "sklearn", "artificial intelligence", "model development",
    ],
    "data_cleaning": [
        "data cleaning", "data cleansing", "cleansing", "cleanse", "data preparation",
        "data prep", "data wrangling", "data quality", "data validation",
        "data integrity", "data profiling",
    ],
    "etl": [
        "etl", "elt", "data pipeline", "data pipelines", "pipelines", "ingestion",
        "data ingestion", "data warehouse", "data warehousing", "data lake",
        "ssis", "informatica", "airflow", "kafka", "spark", "snowflake", "redshift",
        "hadoop", "databricks", "big data", "data mart",
    ],
    "data_modelling": [
        "data model", "data models", "data modelling", "data modeling",
        "dimensional modelling", "dimensional modeling", "star schema", "kimball",
        "entity relationship", "entity-relationship", "erwin", "schema design",
    ],
    "cloud": [
        "aws", "amazon web services", "azure", "gcp", "google cloud",
        "cloud platform", "cloud platforms", "cloud environment", "databricks",
        "snowflake", "s3", "ec2", "sagemaker", "azure synapse", "azure data factory",
    ],
    "stakeholder_comm": [
        "stakeholder", "stakeholders", "stakeholder management",
        "stakeholder engagement", "presenting", "presentation", "presentations",
        "communication skills", "communicate", "communicating", "data storytelling",
        "storytelling", "non-technical", "non technical",
    ],
    "ethics_governance": [
        "gdpr", "data protection", "data governance", "data privacy", "data ethics",
        "information confidentiality",
    ],
}

# Terms that trigger a lexicon match but should not count as a skill mention.
# Derived from annotation guidelines v2.0 Section 3.1 (homonyms and product names).
# Used by the lexicon-based methods to suppress known false positives.
NEGATIVE_PATTERNS = {
    "excel": [r"excellen\w*", r"excel(?:ling|led)? (?:in|at|within)"],
    "reporting": [r"report(?:s|ing)? (?:to|into|directly to)\b", r"direct reports?\b"],
    "etl": [r"demand pipeline", r"(?:drug|product|sales|candidate) pipeline"],
    "programming": [r"clinical coding", r"wellbeing programme", r"programme\b"],
    "cloud": [r"sales cloud", r"service cloud", r"oracle cloud services"],
    "sql": [r"oracle (?:fusion|epm|epbcs|pbcs)", r"hyperion"],
}

PATHS = {
    "corpus": "uk_analyst_corpus_v4_clean.csv",
    "gold_workbook": "gold_standard_annotation_workbook_v2.xlsx",
}
