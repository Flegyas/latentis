from pathlib import Path

from peewee import BooleanField, CharField, Check, DateTimeField, FloatField, ForeignKeyField, Model, SqliteDatabase

from latentis import PROJECT_ROOT

NEXUS_DIR: Path = PROJECT_ROOT / "nexus"
NEXUS_DIR.mkdir(exist_ok=True)

DB_DIR = NEXUS_DIR / "db.sqlite3"

db = SqliteDatabase(str(DB_DIR), pragmas={"foreign_keys": 1})


class Dataset(Model):
    name = CharField()
    split = CharField()
    x_feature = CharField()
    y_feature = CharField()

    class Meta:
        database = db
        indexes = ((("name", "split", "x_feature", "y_feature"), True),)


class DatasetProperty(Model):
    dataset_id = ForeignKeyField(Dataset, backref="properties", null=False)
    name = CharField()
    value = CharField()
    value_type = CharField()

    class Meta:
        database = db


class Space(Model):
    itemclass = CharField()
    timestamp = DateTimeField()
    # encoder = ForeignKeyField(Model, backref="encodings", null=False)
    # dataset = ForeignKeyField(Dataset, backref="encodings", null=False)

    class Meta:
        database = db


class SpaceProperty(Model):
    spaceid = ForeignKeyField(Space, backref="properties", null=False)
    name = CharField()
    value = CharField()
    value_type = CharField()

    class Meta:
        database = db


class EstimationStage(Model):
    # itemclass = CharField()
    x = ForeignKeyField(Space, backref=None, null=True)
    y = ForeignKeyField(Space, backref=None, null=True)
    estimator_key = CharField()  # links to the specific Python Estimator class

    class Meta:
        database = db


class EstimationStageProperty(Model):
    estimationstageid = ForeignKeyField(EstimationStage, backref="estimatorproperties", null=False)
    name = CharField()
    value = CharField()
    value_type = CharField()

    class Meta:
        database = db


class NNModel(Model):
    is_encoder = BooleanField()
    is_decoder = BooleanField()
    fit_data = ForeignKeyField(Dataset, backref=None, null=True)
    test_data = ForeignKeyField(Dataset, backref=None, null=True)

    class Meta:
        database = db
        constraints = [Check("is_encoder OR is_decoder")]


class NNModelProperty(Model):
    nnmodelid = ForeignKeyField(NNModel, backref="properties", null=True)
    name = CharField()
    value = CharField()
    value_type = CharField()

    class Meta:
        database = db


class Correspondence(Model):
    xdata = ForeignKeyField(Dataset, backref="xcorrespondences", null=False)
    ydata = ForeignKeyField(Dataset, backref="ycorrespondences", null=False)
    split = CharField()

    class Meta:
        database = db


class CorrespondenceProperty(Model):
    correspondenceid = ForeignKeyField(Correspondence, backref="properties", null=False)
    name = CharField()
    value = CharField()
    value_type = CharField()

    class Meta:
        database = db


class Experiment(Model):
    x_fit = ForeignKeyField(Space, backref=None, null=False)
    y_fit = ForeignKeyField(Space, backref=None, null=False)
    x_test = ForeignKeyField(Space, backref=None, null=False)
    y_test = ForeignKeyField(Space, backref=None, null=True)
    estimation_stage = ForeignKeyField(EstimationStage, backref="results", null=False)
    fit_correspondence = ForeignKeyField(Correspondence, backref=None, null=False)
    test_correspondence = ForeignKeyField(Correspondence, backref=None, null=False)
    y_fit_decoder = ForeignKeyField(NNModel, backref="experiments", null=True)
    measure = CharField()
    measure_value = FloatField()
    measure_type = CharField()

    class Meta:
        database = db


db.connect()
db.create_tables(
    [
        Space,
        SpaceProperty,
        Dataset,
        DatasetProperty,
        Correspondence,
        CorrespondenceProperty,
        Experiment,
        NNModel,
        NNModelProperty,
        EstimationStage,
        EstimationStageProperty,
    ]
)

# dataset_id = Dataset(
#     name="mnist",
#     split="train",
#     x_feature="image",
#     y_feature="label",
# ).save()

# DatasetProperty(
#     dataset_id=dataset_id,
#     name="image_shape",
#     value="(28, 28)",
#     value_type="tuple",
# ).save()

DatasetProperty(
    dataset_id="dataset_id",
    name="image_shape",
    value="(28, 28)",
    value_type="tuple",
).save()

# def initindex(path: Path, itemclass: Type) -> None:
#     try:
#         index = DiskIndex.loadfromdisk(path=path)
#     except FileNotFoundError:
#         index = DiskIndex(rootpath=path, itemclass=itemclass)
#         index.savetodisk()

#     return index


# correspondencesindex: DiskIndex = initindex(path=NEXUSDIR / "correspondences", itemclass=Correspondence)
# spaceindex: DiskIndex = initindex(path=NEXUSDIR / "spaces", itemclass=LatentSpace)
# decodersindex = initindex(path=NEXUSDIR / "decoders", itemclass=LatentisModule)
