#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    logger.info("Starting step: basic_cleaning")
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact.
    # Log what it's being done
    logger.info(f"Downloading artifact {args.input_artifact}")

    # Define the artifact path for the input
    artifact_path = run.use_artifact(args.input_artifact).file()

    # Read the csv file and leave it into a dataframe
    logger.info("Reading the downloaded file")
    df = pd.read_csv(artifact_path)

    # Drop outliers
    logger.info("Dropping outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Feature engineering")
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    # Discard out of range values
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the dataframe to csv
    logger.info("Saving new file clean_sample.csv")
    df.to_csv('clean_sample.csv', index=False)

    # Create the artifact object for the output
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    # Add the csv file created to the artifact object
    artifact.add_file("clean_sample.csv")

    # Log it into wandb
    logger.info("Logging this new file to W&B")
    run.log_artifact(artifact)
    logger.info("Finished step: basic_cleaning")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="The name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="The name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="The type for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="The minimum price to consider",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="The maximum price to consider",
        required=True
    )


    args = parser.parse_args()

    go(args)
