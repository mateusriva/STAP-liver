#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.utilities.io

assortment of input/output utilities for the project

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Apr 24, 2018
"""

from __future__ import print_function, division

from glob import glob
import os

import nibabel as nib
import pydicom


def split_filename(filepath):
    """ split a filepath into the full path, filename, and extension (works with .nii.gz) """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def open_nii(filepath):
    """ open a nifti file with nibabel and return the object """
    image = os.path.abspath(os.path.expanduser(filepath))
    obj = nib.load(image)
    return obj

def open_dcm(filepath):
    """ open a dicom file with nibabel and return the object """
    image = os.path.abspath(os.path.expanduser(filepath))
    obj = pydicom.read_file(image)
    return obj

def save_nii(obj, outfile, data=None, is_nii=False):
    """ save a nifti object """
    if not is_nii:
        if data is None:
            data = obj.get_data()
        nib.Nifti1Image(data, obj.affine, obj.header)\
            .to_filename(outfile)
    else:
        obj.to_filename(outfile)

def save_dcm(obj, outfile):
    """ save a dicom object """
    obj.save_as(outfile)


def glob_nii(dir):
    """ return a sorted list of nifti files for a given directory """
    fns = sorted(glob(os.path.join(dir, '*.nii*')))
    return fns

def glob_dcm(dir):
    """ return a sorted list of dicom files for a given directory """
    fns = sorted(glob(os.path.join(dir, '*.dcm')))
    return fns
