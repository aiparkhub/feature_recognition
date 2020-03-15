# -*- coding:utf-8 -*-
#
# Geek International Park | 极客国际公园 & GeekParkHub | 极客实验室
# Website | https://www.geekparkhub.com
# Description | Open · Creation |
# Open Source Open Achievement Dream, GeekParkHub Co-construction has never been seen before.
#
# HackerParkHub | 黑客公园
# Website | https://www.hackerparkhub.org
# Description | In the spirit of fearless exploration, create unknown technology and worship of technology.
#
# AIParkHub | 人工智能公园
# Website | https://github.com/aiparkhub
# Description | Embark on the wave of AI and push the limits of machine intelligence.
#
# @GeekDeveloper : JEEP-711
# @Author : system
# @Version : 0.2.5
# @Program : 人像识别 命令行工具 | Portrait recognition command line tool
# @File : face_recognition_cli.py
# @Description : 人像识别 命令行工具 | Portrait recognition command line tool
# @Copyright © 2019 - 2020 AIParkHub-Organization. All rights reserved.


# 导入 第三方模块 | Import third-party modules
from __future__ import print_function
import PIL.Image
import numpy as np

# 导入 标准库&内置模块 | Import standard library & built-in modules
import click
import os
import re
import multiprocessing
import itertools
import sys

# 导入 自定义模块 | Import Custom Module
import face_recognition.api as face_recognition


def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 1:
            click.echo(
                "WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            click.echo(
                "WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])

    return known_names, known_face_encodings


def print_result(filename, name, distance, show_distance=False):
    if show_distance:
        print("{},{},{}".format(filename, name, distance))
    else:
        print("{},{}".format(filename, name))


def test_image(
        image_to_check,
        known_names,
        known_face_encodings,
        tolerance=0.6,
        show_distance=False):
    unknown_image = face_recognition.load_image_file(image_to_check)

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(
            known_face_encodings, unknown_encoding)
        result = list(distances <= tolerance)

        if True in result:
            [print_result(image_to_check,
                          name,
                          distance,
                          show_distance) for is_match,
             name,
             distance in zip(result,
                             known_names,
                             distances) if is_match]
        else:
            print_result(image_to_check, "unknown_person", None, show_distance)

    if not unknown_encodings:
        # print out fact that no faces were found in image
        print_result(image_to_check, "no_persons_found", None, show_distance)


def image_files_in_folder(folder):
    return [
        os.path.join(
            folder,
            f) for f in os.listdir(folder) if re.match(
            r'.*\.(jpg|jpeg|png)',
            f,
            flags=re.I)]


def process_images_in_process_pool(
        images_to_check,
        known_names,
        known_face_encodings,
        number_of_cpus,
        tolerance,
        show_distance):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use
    # 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance)
    )

    pool.starmap(test_image, function_parameters)


@click.command()
@click.argument('known_people_folder')
@click.argument('image_to_check')
@click.option(
    '--cpus',
    default=1,
    help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
@click.option(
    '--tolerance',
    default=0.6,
    help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
@click.option(
    '--show-distance',
    default=False,
    type=bool,
    help='Output face distance. Useful for tweaking tolerance setting.')
def main(known_people_folder, image_to_check, cpus, tolerance, show_distance):
    known_names, known_face_encodings = scan_known_people(known_people_folder)

    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo(
            "WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1

    if os.path.isdir(image_to_check):
        if cpus == 1:
            [test_image(image_file, known_names, known_face_encodings, tolerance, show_distance)
             for image_file in image_files_in_folder(image_to_check)]
        else:
            process_images_in_process_pool(
                image_files_in_folder(image_to_check),
                known_names,
                known_face_encodings,
                cpus,
                tolerance,
                show_distance)
    else:
        test_image(
            image_to_check,
            known_names,
            known_face_encodings,
            tolerance,
            show_distance)


if __name__ == "__main__":
    main()
