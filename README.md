## Image Quality Evaluation (JPEG, WEBP, AVIF and HEIFs)
[![Build and publish Docker image](https://github.com/sumn2u/image_quality_evaluation/actions/workflows/main.yml/badge.svg)](https://github.com/sumn2u/image_quality_evaluation/actions/workflows/main.yml)

Image Quality Evaluation of different image file formats (JPEG, WEBP, AVIF and HEIFs)

## VERSION

1.0

## AUTHOR

Suman Kunwar <suman@dwaste.live>

## DATASETS

- CBSD68: contains 68 images in bmp format (colored) 
- Kodak24: 24 images, colored png format, various sizes
- McM: 18 images in tif format (colored) 
- Set12: 12 images png fromat (lena discontiued)
- Set68: 68 images png (8-bit grayscale BMP) fix on 

## CONTENTS


directory | description

    image_quality_evaluation
    ├── images                  # folder that will contain the converted images 
    ├── templates               # template file to render content in web 
    ├── templates               # template file to render content in web 
    ├── Dockerfile              # dockerfile to run application
    ├── environment.yaml        # environment configuration file for Miniconda used in docker
    ├── image_processing.py     # code to convert the file and measure its quality 
    ├── main.py                 # main application file
    ├── requirements.txt        # dependencis packages/libraries   
    └── ...

## REQUIREMENTS

The application is containerized using docker 🐳 for development purpose.

Download [Docker Desktop](https://www.docker.com/products/docker-desktop) for Mac or Windows. [Docker Compose](https://docs.docker.com/compose) will be automatically installed. On Linux, make sure you have the latest version of [Compose](https://docs.docker.com/compose/install/).


## COMPILATION INSTRUCTIONS

Simply run the following command to start the application. 

```shell
docker build --tag image_quality_evaluation . # this will run create a docker image of the application 
docker run -p 5000:5000 image_quality_evaluation #this will run the application and maps local machine 5000 port to docker's 5000 port
```

Once completed, we can access the application by visiting <http://localhost:5000> on our browser.


## EXAMPLES OF USE

We can upload the png image from file upload. It will then convert the file into `JPEG, WEBP, AVIF and HEIF` formats. From the output fomats it quality metrics such as `MSE, PSNR and SSIM` to compare the results. The loading time is also compared to render the image.

One can use the cameraman.png attached in this project to test the results.

> **Note**
> HEIF image are not supported in most of the [browsers](https://caniuse.com/?search=heif) so one might see broken images for unsupported browser.

## COPYRIGHT AND LICENSE INFORMATION

Copyright (c) 2023 Suman Kunwar

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.
