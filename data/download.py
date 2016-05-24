#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import urllib
import os
import zipfile

url = 'http://sci2s.ugr.es/keel/dataset/data/'
output_path = './data'

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
    if not os.path.exists(output_path + '/classification'):
        os.mkdir(output_path + '/classification')
    if not os.path.exists(output_path + '/regression'):
        os.mkdir(output_path + '/regression')

    for item in download_list:
        if not os.path.exists(output_path + '/' + item):
            print 'downloading ' + item + '...'
            full_url = url + item
            urllib.urlretrieve(full_url, output_path + '/' + item)

def extract():
    data_path = []
    for item in download_list:
        f = open(output_path + '/' + item, 'rb')
        z = zipfile.ZipFile(f)
        for filename in z.namelist():
            z.extract(filename, output_path)
            data_path.append(output_path + '/' + filename)
        f.close()
    return data_path

if __name__ == '__main__':
    download()
    extract()

