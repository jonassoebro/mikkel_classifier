# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from process_raw_image import process_raw_image

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # Make target dirs if they do not already exists
    if not os.path.isdir("data/processed/negative"):
        os.mkdir("data/processed/negative")
    if not os.path.isdir("data/processed/positive"):
        os.mkdir("data/processed/positive")
    
    # Rename images as numbers
    counter = 0
    for img_name in os.listdir(input_filepath):
        img_path = os.path.join(input_filepath, img_name)
        process_raw_image(img_path, output_filepath, output_imgname=counter, imgsize=64)
        counter = counter+1
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
