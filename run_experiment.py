#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('./data')

import download
import re
import subprocess
import os

executable = './test/ModelTest'

params = {
    'iris': {
        'nhidden': 3,
        'h0': {
            'lrate': 0.01,
            'lambda': 1e-6,
            'nodes': 6,
            'act': 'arctan'
        },
        'h1': {
            'lrate': 0.01,
            'lambda': 1e-6,
            'nodes': 6,
            'act': 'relu'
        },
        'h2': {
            'lrate': 0.01,
            'lambda': 1e-6,
            'nodes': 6,
            'act': 'sigmoid'
        },
        'output': {
            'cost': 'softmax',
            'lrate': 0.01,
            'lambda': 1e-6
        },
        'trainer': {
            'name': 'gd',
            'max_iterations': 300000,
            'r0': 0.01,
            'k': 1e-4,
            'step': 300000
        },
        'trail': 3
    },
    'nursery': {
        'nhidden': 4,
        'h0': {
            'lrate': 1e1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'h1': {
            'lrate': 1e1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'h2': {
            'lrate': 1e-1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'h3': {
            'lrate': 1e-1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'output': {
            'cost': 'softmax',
            'lrate': 1e-1,
            'lambda': 0
        },
        'trainer': {
            'name': 'sgd',
            'max_iterations': 1000,
            'r0': 1e-1,
            'k': 1e-8,
            'step': 1000
        },
        'trail': 1
    },
    'movement_libras': {
        'nhidden': 4,
        'h0': {
            'lrate': 1e1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'h1': {
            'lrate': 1e1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'h2': {
            'lrate': 1e-1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'h3': {
            'lrate': 1e-1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'output': {
            'cost': 'softmax',
            'lrate': 1e-1,
            'lambda': 0
        },
        'trainer': {
            'name': 'sgd',
            'max_iterations': 1000,
            'r0': 1e-1,
            'k': 1e-8,
            'step': 1000
        },
        'trail': 1
    },
    'phoneme': {
        'nhidden': 4,
        'h0': {
            'lrate': 1e1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'h1': {
            'lrate': 1e1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'h2': {
            'lrate': 1e-1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'h3': {
            'lrate': 1e-1,
            'lambda': 0,
            'nodes': 10,
            'act': 'sigmoid'
        },
        'output': {
            'cost': 'softmax',
            'lrate': 1e-1,
            'lambda': 0
        },
        'trainer': {
            'name': 'sgd',
            'max_iterations': 1000,
            'r0': 1e-1,
            'k': 1e-8,
            'step': 1000
        },
        'trail': 1
    },
    'zoo': {
        'nhidden': 2,
        'h0': {
            'lrate': 1e-1,
            'lambda': 1e-3,
            'nodes': 10,
            'act': 'arctan'
        },
        'h1': {
            'lrate': 1e-1,
            'lambda': 1e-3,
            'nodes': 10,
            'act': 'arctan'
        },
        'output': {
            'cost': 'softmax',
            'lrate': 1e-1,
            'lambda': 1e-5
        },
        'trainer': {
            'name': 'sgd',
            'max_iterations': 1000,
            'r0': 1e-3,
            'k': 1e-8,
            'step': 1000
        },
        'trail': 5
    },
}

def isnum(string):
    result = re.match('-?[0-9.]+', string)
    if result:
        return True
    return False

def isallnum(strings):
    for string in strings:
        if not isnum(string):
            return False
    return True

def getn(training_data):
    return len(training_data[0]) - 1

def geto(training_data):
    record = {}
    last = len(training_data[0]) - 1
    for entry in training_data:
        if not entry[last] in record:
            record[entry[last]] = 1
    return len(record.keys())

def max_label(training_data):
    max_val = 0
    for sample in training_data:
        if sample[-1] > max_val:
            max_val = sample[-1]
    return max_val


def replace(training_data, index, old, new):
    for i in range(len(training_data)):
        if training_data[i][index] == old:
            training_data[i][index] = new
    return training_data

if __name__ == '__main__':
    download.download()
    data_path = download.extract()

    if len(sys.argv) == 1:
        exit(0)

    for data in data_path['c']:
        name = data.split('/')[-1][:-4]

        if name != sys.argv[1]:
            continue

        print 'model for ' + name
        f = open(data, 'r')
        content = []
        for line in f:
            if not line.startswith('@'):
                content.append(line.replace(' ', '').split(','))

        training_data = [[0 for i in range(len(content[0]))] for c in content]
        for i in range(len(content[0])):
            if isallnum([x[i] for x in content]):
                for j in range(len(training_data)):
                    training_data[j][i] = float(content[j][i])
            else:
                mapping = {}
                num = 0
                for j in range(len(training_data)):
                    if not content[j][i] in mapping:
                        mapping[content[j][i]] = num
                        num = num + 1
                    training_data[j][i] = mapping[content[j][i]]


        n = str(getn(training_data))
        o = str(geto(training_data))
        m = str(len(training_data))

        if int(max_label(training_data)) == geto(training_data):
            training_data = replace(training_data, getn(training_data),
                                    max_label(training_data), 0)

        for line in training_data:
            print line

        print 'n = ', n
        print 'o = ', o
        print 'm = ', m

        input_data = n + ' ' + o + ' ' + m + '\n'
        if os.path.exists(executable):
            p = subprocess.Popen('./test/ModelTest', stdin=subprocess.PIPE)
            for sample in training_data:
                for entry in sample:
                    input_data = input_data + (str(entry) + ' ')
                input_data = input_data + '\n'

            if name in params:
                param = params[name]
                # hidden layer params
                nhlayer = param['nhidden']
                input_data = input_data + str(nhlayer) + '\n'

                for k in range(nhlayer):
                    hlayer_param = param['h'+str(k)]
                    lrate = hlayer_param['lrate']
                    lb = hlayer_param['lambda']
                    nnodes = hlayer_param['nodes']
                    actfunc = hlayer_param['act']

                    input_data = input_data + str(lrate) + '\n'
                    input_data = input_data + str(lb) + '\n'
                    input_data = input_data + str(nnodes) + '\n'
                    input_data = input_data + actfunc + '\n'

                # output layer params
                input_data = input_data + param['output']['cost'] + '\n'
                input_data = input_data + str(param['output']['lrate']) + '\n'
                input_data = input_data + str(param['output']['lambda']) + '\n'

                # trainer params
                input_data = input_data + param['trainer']['name'] + '\n'
                input_data = input_data + \
                        str(param['trainer']['max_iterations']) + '\n'
                input_data = input_data + str(param['trainer']['r0']) + '\n'
                input_data = input_data + str(param['trainer']['k']) + '\n'
                input_data = input_data + str(param['trainer']['step']) + '\n'

                # trail
                input_data = input_data + str(param['trail']) + '\n'

                # print input_data[:10]
                p.communicate(input=input_data)
            else:
                print 'no parameter setup yet!!'

        else:
            print 'build the project first!!'
        break

