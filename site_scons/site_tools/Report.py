#
# @Author : Jean-Pascal Mercier <jean-pascal.mercier@agsis.com>
#
# @Copyright (C) 2010 Jean-Pascal Mercier
#
# All rights reserved.
#
__doc__ = """
"""

from SCons.Script import Builder, Mkdir, Copy, Execute
import sys

import matplotlib
matplotlib.use('Agg')

header = \
"""
<html>
<head>
    <link href="%s" rel="stylesheet" type="text/css">
</head>
<body>
    <h1> Advanced GeoScience Imaging Solution Ltd. </h1>
"""

footer = \
"""
    <div id="footer">
        <h3> Created %s with Lotic</h3> &copy %d AGSIS Ltd. all rights reserved
    </div>
    </body>
</html>
"""

adverbs = ['smootly',
           'easily',
           'indubitably',
           'as intended',
           'pridefully'
           ]


def generate(env):
    import time
    import hashlib
    import os
    import datetime
    import pickle
    import numpy as np

    import matplotlib.pyplot as plt

    import string

    element_string = \
"""
<div class="element">
    <h3> ${target} </h3> <h3> Date : </h3> ${date}
    <p>
    ${rest}
    </p>
</div>
"""
    element_template = string.Template(element_string)


    def BuildReportEmitter(target, source, env):
        return [os.path.join(env['REPORT_ROOT'], 'index.html')], source


    def ReportEmitter(target, source, env):
        basename = hashlib.md5(str(source[0])).hexdigest()
        return [os.path.join(env['REPORT_ROOT'], basename) + ".html"], source

    def Report(source, target, env):
        filename = str(target[0])
        timestamp = source[0].get_timestamp()

        dtime = datetime.datetime.fromtimestamp(timestamp)

        html = element_template.substitute(target = str(source[0]),
                                           date = str(dtime),
                                           rest = "")

        with open(filename, 'w') as f:
            f.write(html)

    def ImageReportEmitter(target, source, env):
        basename = hashlib.md5(str(source[0])).hexdigest()
        htmlfile = os.path.join(env['REPORT_ROOT'], basename) + ".html"
        imagefile = os.path.join(env['REPORT_ROOT'], 'img', basename) + ".png"
        return [htmlfile, imagefile], source

    def HistogramReport(source, target, env):
        filename = str(target[0])
        timestamp = source[0].get_timestamp()
        histofile = str(target[1])

        dtime = datetime.datetime.fromtimestamp(timestamp)

        imgline = '<img src="%s" alt="Histogram"/>' %\
                histofile.replace(os.path.dirname(filename), "")[1:]

        column = np.load(str(source[0]))

        mean = np.average(column)
        stdev = np.std(column)
        abs_stdev = np.std(abs(column))

        stats = dict(Average = str(mean), Stdev = str(stdev))


        statline = "".join(["<h3>%s :</h3> %s" % (k, v) for (k, v) in stats.iteritems()])


        html = element_template.substitute(target = str(source[0]),
                                           date = str(dtime),
                                           rest = imgline + statline)

        with open(filename, 'w') as f:
            f.write(html)

        size = env['REPORT_HISTOGRAM_SIZE']

        f = plt.figure(figsize = size)
        ax = f.add_subplot(1,1,1)

        hrange = source[1].value

        ax.hist(column, range = hrange,
                        bins = env['REPORT_HISTOGRAM_BINS'],
                        normed = True,
                        facecolor = env['REPORT_HISTOGRAM_FACECOLOR'],
                        alpha = env['REPORT_HISTOGRAM_ALPHA'])

        x = np.linspace(hrange[0], hrange[1])
        y = (1.0 / np.sqrt(2 * np.pi * (stdev ** 2))) *\
                np.exp(-0.5 * ((x - mean) ** 2) / stdev ** 2)

        ax.plot(x, y, 'b--')


        f.savefig(histofile)


    def ResidualHistogramAction(source, target, env):
        filename = str(target[0])
        histofile = str(target[1])
        timestamp = source[0].get_timestamp()

        dtime = datetime.datetime.fromtimestamp(timestamp)

        imgline = '<img src="%s" alt="Histogram"/>' %\
                histofile.replace(os.path.dirname(filename), "")[1:]

        column = []
        for s in source:
            r = np.load(str(s))[env['REPORT_RESIDUAL_HISTOGRAM_COLUMN']]
            if r is not None:
                column.extend(r)
        column = np.array(column)

        mean = np.average(column)
        stdev = np.std(column)
        abs_stdev = np.std(abs(column))

        stats = dict(Average = str(mean), Stdev = str(stdev))


        statline = "".join(["<h3>%s :</h3> %s" % (k, v) for (k, v) in stats.iteritems()])


        html = element_template.substitute(target = str(source[0]),
                                           date = str(dtime),
                                           rest = imgline + statline)

        with open(filename, 'w') as f:
            f.write(html)

        size = env['REPORT_HISTOGRAM_SIZE']

        f = plt.figure(figsize = size)
        ax = f.add_subplot(1,1,1)

        hrange = env['REPORT_RESIDUAL_HISTOGRAM_RANGE']

        ax.hist(column, range = hrange,
                        bins = env['REPORT_HISTOGRAM_BINS'],
                        normed = True,
                        facecolor = env['REPORT_HISTOGRAM_FACECOLOR'],
                        alpha = env['REPORT_HISTOGRAM_ALPHA'])

        x = np.linspace(hrange[0], hrange[1])
        y = (1.0 / np.sqrt(2 * np.pi * (stdev ** 2))) *\
                np.exp(-0.5 * ((x - mean) ** 2) / stdev ** 2)

        ax.plot(x, y, 'b--')


        f.savefig(histofile)



    def InversionReportAction(source, target, env):
        print(source[0], [str(t) for t in target])
        statfile = str(source[0])
        filename = str(target[0])
        graphfile = str(target[1])

        invstats = pickle.load(open(statfile, 'rb'))

        timestamp = source[0].get_timestamp()
        size = env['REPORT_HISTOGRAM_SIZE']

        dtime = datetime.datetime.fromtimestamp(timestamp)
        imgline = '<img src="%s" alt="Inversion"/>' %\
                graphfile.replace(os.path.dirname(filename), "")[1:]


        html = element_template.substitute(target = str(source[0]),
                                           date = str(dtime),
                                           rest = imgline)
        print(filename)
        with open(filename, 'w') as f:
            f.write(html)

        f = plt.figure(figsize = size)
        ax = f.add_subplot(1,1,1)
        ax.plot(invstats, 'g--')

        f.savefig(graphfile)


    def ConcatenateReport(source, target, env):
        target = str(target[0])
        cssroot = os.path.join(os.path.dirname(target), 'css')
        Execute(Mkdir(cssroot))
        stylesheet = os.path.join(cssroot, os.path.basename(str(source[0])))
        Execute(Copy(stylesheet, source[0]))

        with open(target, 'w') as f:
            f.write(header % os.path.join('css', os.path.basename(stylesheet)))
            for s in source[1:]:
                f.write(open(str(s)).read())
            footnote = footer % \
                    (adverbs[np.random.randint(0, len(adverbs))], time.localtime().tm_year)
            f.write(footnote)

    env['BUILDERS']['ConcatenateReport'] = Builder(action = ConcatenateReport,
                                                   emitter = BuildReportEmitter)
    env['BUILDERS']['HistogramReport'] = Builder(action = HistogramReport,
                                                 emitter = ImageReportEmitter)

    env['BUILDERS']['ResidualHistogramReport'] = Builder(action = ResidualHistogramAction,
                                                         emitter = ImageReportEmitter)
    env['BUILDERS']['Report'] = Builder(action = Report, emitter = ReportEmitter)
    env['BUILDERS']['InversionReport'] = Builder(action = InversionReportAction,
                                                 emitter = ImageReportEmitter)

    env['REPORT_HISTOGRAM_SIZE'] = (9, 2)
    env['REPORT_HISTOGRAM_BINS'] = 100
    env['REPORT_HISTOGRAM_FACECOLOR'] = 'green'
    env['REPORT_HISTOGRAM_ALPHA'] = 0.7
    env['REPORT_RESIDUAL_HISTOGRAM_RANGE'] = [-1, 1]
    env['REPORT_RESIDUAL_HISTOGRAM_COLUMN'] = 'residual'


def exists(env):
    return 1


if __name__ == '__main__':
    from twisted.web import server, resource
    from twisted.internet import reactor
    import glob
    import os

    siteroot = sys.argv[1]
    divs = [open(f).read() for f in glob.glob(os.path.join(siteroot, "*")) if not os.path.isdir(f)]


    class Simple(resource.Resource):
        isLeaf = True
        def render_GET(self, request):
            print(request.uri)
            if request.uri == "/":
                return header + "\n".join(divs) + footer
            else:
                return open(os.path.join(sys.argv[1], request.uri[1:])).read()


    PORT = 8000

    site = server.Site(SimpleServer())
    reactor.listenTCP(PORT, site)
    reactor.run()
