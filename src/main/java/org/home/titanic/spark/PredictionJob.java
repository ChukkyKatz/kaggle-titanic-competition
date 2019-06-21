package org.home.titanic.spark;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import static org.apache.spark.sql.types.DataTypes.*;
import static org.apache.spark.sql.functions.*;

public class PredictionJob {

    private SparkSession spark;

    public void start() {

        spark = SparkSession
                .builder()
                .appName("titanic-prediction")
                .master("local[*]")
                .getOrCreate();

        final var trainDf = readData("data\\train.csv", new StructType(new StructField[] {
                new StructField("PassengerId", IntegerType, false, Metadata.empty()),
                new StructField("Survived", IntegerType, false, Metadata.empty()),
                new StructField("Pclass", IntegerType, false, Metadata.empty()),
                new StructField("Name", StringType, false, Metadata.empty()),
                new StructField("Sex", StringType, false, Metadata.empty()),
                new StructField("Age", DoubleType, true, Metadata.empty()),
                new StructField("SibSp", IntegerType, false, Metadata.empty()),
                new StructField("Parch", IntegerType, false, Metadata.empty()),
                new StructField("Ticket", StringType, false, Metadata.empty()),
                new StructField("Fare", DoubleType, true, Metadata.empty()),
                new StructField("Cabin", StringType, true, Metadata.empty()),
                new StructField("Embarked", StringType, true, Metadata.empty())
        }));

        final var testDf = readData("data\\test.csv", new StructType(new StructField[] {
                new StructField("PassengerId", IntegerType, false, Metadata.empty()),
                new StructField("Pclass", IntegerType, false, Metadata.empty()),
                new StructField("Name", StringType, false, Metadata.empty()),
                new StructField("Sex", StringType, false, Metadata.empty()),
                new StructField("Age", DoubleType, true, Metadata.empty()),
                new StructField("SibSp", IntegerType, false, Metadata.empty()),
                new StructField("Parch", IntegerType, false, Metadata.empty()),
                new StructField("Ticket", StringType, false, Metadata.empty()),
                new StructField("Fare", DoubleType, true, Metadata.empty()),
                new StructField("Cabin", StringType, true, Metadata.empty()),
                new StructField("Embarked", StringType, true, Metadata.empty())
        }));

        // Combining test and train as single to apply some function
        final var allDataDf = trainDf.drop("Survived")
                .union(testDf);

        final var filledMissingDatadDf = fillMissingValues(allDataDf);

        final var extraFeaturesTrainDf = extractFeatures(filledMissingDatadDf);

        final var preparedTrainDf = trainDf.join(extraFeaturesTrainDf,"PassengerId")
                .select("PassengerId", "Survived", "pclass_f", "sibsp_f", "parch_f", "family_size_f",
                        "title_f", "sex_f", "age_f", "fare_f", "embarked_f")
                .withColumn("survived_l", col("Survived").cast(DoubleType));

        final var preparedTestDf = testDf.join(extraFeaturesTrainDf,"PassengerId")
                .select("PassengerId", "pclass_f", "sibsp_f", "parch_f", "family_size_f",
                        "title_f", "sex_f", "age_f", "fare_f", "embarked_f");

        final var victorizedFeaturesTrainDf = featureVectorPrepare(preparedTrainDf);
        final var victorizedFeaturesTestDf = featureVectorPrepare(preparedTestDf);

        // Model training
        final RandomForestClassifier classifier = new RandomForestClassifier()
                .setFeaturesCol("features")
                .setLabelCol("survived_l")
                .setNumTrees(10);

        final RandomForestClassificationModel model = classifier.fit(victorizedFeaturesTrainDf);

        final var predictions = model.transform(victorizedFeaturesTestDf);

        savePredictions(predictions);
        predictions.show(1500);

        spark.stop();
    }

    private Dataset<Row> readData(final String path, final StructType schema) {
        return spark.read()
                .option("header", true)
                .schema(schema)
                .csv(path);
    }

    private Dataset<Row> fillMissingValues(final Dataset<Row> df) {

        // "Cabin" column has 75% of missing data in both Test and Train data sets so we can drop it
        // "Ticket" importance is doubtful and need to be investigated more deeply, currently we don't use it
        final var droppedCabinDf = df.drop("Cabin").drop("Ticket");

        // Filling missed values in "Embarked" column by mode value of this column (only 2 values)
        final String embarkedMode = droppedCabinDf.select("Embarked")
                .groupBy("Embarked")
                .count()
                .orderBy(desc("count"))
                .first()
                .getString(0);

        final var embarkedFilledDf = droppedCabinDf.na()
                .fill(embarkedMode, new String[]{"Embarked"});

        // Filling missed values in "Fare" column by median value of this column
        final var sortedIndexedFare = embarkedFilledDf.select("Fare")
                .orderBy("Fare")
                .withColumn("id", row_number()
                        .over(Window.orderBy("Fare")));

        final long countsFareMedian = sortedIndexedFare.count() / 2;

        final double fareMedian = sortedIndexedFare.select("Fare")
                .where("id = " + countsFareMedian)
                .first()
                .getDouble(0);

        final var fareFilledDf = embarkedFilledDf.na()
                .fill(fareMedian, new String[]{"Fare"});

        // Filling missed values in "Age" column by median value of this column
        final var sortedIndexedAgeDf = fareFilledDf.select("Age")
                .orderBy("Age")
                .withColumn("id", row_number()
                        .over(Window.orderBy("Age")));

        final long countsAgeMedian = sortedIndexedAgeDf.count() / 2;

        final double ageMedian = sortedIndexedAgeDf.select("Age")
                .where("id = " + countsAgeMedian)
                .first()
                .getDouble(0);

        return fareFilledDf.na()
                .fill(ageMedian, new String[]{"Age"});
    }

    private Dataset<Row> extractFeatures(final Dataset<Row> df) {

        // Calculating family size for all passengers
        final var allDataFamilyDf = df.withColumn("FamilySize",
                col("SibSp")
                        .plus(col("Parch"))
                        .plus(1));

        // Casting all features to double type
        final var castedDf = allDataFamilyDf.withColumn("pclass_f", col("Pclass").cast(DoubleType))
                .withColumn("sibsp_f", col("SibSp").cast(DoubleType))
                .withColumn("parch_f", col("Parch").cast(DoubleType))
                .withColumn("family_size_f", col("FamilySize").cast(DoubleType));

        // Creating a new feature Title, containing the titles of passenger names
        final var allDataTitleDf = castedDf
                .withColumn("Title", regexp_extract(col("Name"),
                        "\\s([A-Za-z]+)\\.", 1))
                .withColumn("Title", regexp_replace(col("Title"),
                "Lady|Countess|Don|Sir|Jonkheer|Dona", "Royalty"))
                .withColumn("Title", regexp_replace(col("Title"),
                        "Capt|Col|Major|Rev|Dr", "Officer"))
                .withColumn("Title", regexp_replace(col("Title"),
                        "Mlle", "Miss"))
                .withColumn("Title", regexp_replace(col("Title"),
                        "Ms", "Miss"))
                .withColumn("Title", regexp_replace(col("Title"),
                        "Mme", "Mrs"));

        final StringIndexer titleIndexer = new StringIndexer()
                .setInputCol("Title")
                .setOutputCol("title_f");

        final var titleIndexedDf = titleIndexer.fit(allDataTitleDf).transform(allDataTitleDf);

        // Indexing "Sex" column to double values
        final StringIndexer sexIndexer = new StringIndexer()
                .setInputCol("Sex")
                .setOutputCol("sex_f");

        final var sexIndexedDf = sexIndexer.fit(titleIndexedDf).transform(titleIndexedDf);

        // Dividing age to categories
        final var allDataAgeBucketizedDf = bucketizeAge(sexIndexedDf);

        // Dividing fare to categories
        final var allDataFareBucketizedDf = bucketizeFare(allDataAgeBucketizedDf);

        // Indexing "Embarked" column to double values
        final StringIndexer embarkedIndexer = new StringIndexer()
                .setInputCol("Embarked")
                .setOutputCol("embarked_f");

        final var embarkedIndexedDf = embarkedIndexer.fit(allDataFareBucketizedDf).transform(allDataFareBucketizedDf);

        return embarkedIndexedDf.drop("Pclass", "Age", "Name", "SibSp",
                "Sex", "Parch", "FamilySize", "Fare", "Embarked", "Title");
    }

    private static Dataset<Row> bucketizeAge(final Dataset<Row> dataset) {
        double[] ageBuckets = {0d, 12d, 20d, 50d, 75d, 99d};
        final Bucketizer ageBucketizer = new Bucketizer()
                .setSplits(ageBuckets)
                .setInputCol("Age")
                .setOutputCol("age_f");

        return ageBucketizer.transform(dataset);
    }

    private static Dataset<Row> bucketizeFare(final Dataset<Row> dataset) {
        double[] fareBuckets = {0d, 7.91d, 14.45d, 31d, 120d, 180d, 515d};
        final Bucketizer ageBucketizer = new Bucketizer()
                .setSplits(fareBuckets)
                .setInputCol("Fare")
                .setOutputCol("fare_f");
        
        return ageBucketizer.transform(dataset);
    }

    private Dataset<Row> featureVectorPrepare(final Dataset<Row> dataset) {
        final VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"pclass_f", "sibsp_f", "parch_f", "family_size_f",
                        "title_f", "sex_f", "age_f", "fare_f", "embarked_f"})
                .setOutputCol("features");

        return vectorAssembler.transform(dataset);
    }

    private void savePredictions(final Dataset<Row> predictions) {
        final StructType solutionSchema = new StructType(new StructField[]{
                new StructField("PassengerId", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("Survived", DataTypes.IntegerType, false, Metadata.empty()),
        });
        final ExpressionEncoder<Row> encoder = RowEncoder.apply(solutionSchema);

        predictions.select("PassengerId", "prediction")
                .map((MapFunction<Row, Row>) row ->
                        RowFactory.create(row.getInt(0), (int) row.getDouble(1)), encoder)
                .coalesce(1).write().option("header", true).csv("output");
    }
}
