#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import urllib
import os
import zipfile

url = 'http://sci2s.ugr.es/keel/dataset/data/'

download_list = [
    'classification/iris.zip',
    'classification/phoneme.zip',
    'classification/nursery.zip',
    'classification/zoo.zip',
    'classification/poker.zip',
    'classification/movement_libras.zip',
    'regression/diabetes.zip',
    'regression/abalone.zip',
    'regression/stock.zip',
    'regression/house.zip'
]

def download():
    if not os.path.exists('classification'):
        os.mkdir('classification')
    if not os.path.exists('regression'):
        os.mkdir('regression')

    for item in download_list:
        if not os.path.exists(item):
            print 'downloading ' + item + '...'
            full_url = url + item
            urllib.urlretrieve(full_url, item)


def extract():
    output_path = './'
    for item in download_list:
        f = open(item, 'rb')
        z = zipfile.ZipFile(f)
        for filename in z.namelist():
            z.extract(filename, output_path)
        f.close()

if __name__ == '__main__':
    download()
    extract()

