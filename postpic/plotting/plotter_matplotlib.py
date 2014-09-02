"""
This package provides the MatplotlibPlotter Class.

This Class can be used to plot Field Objects using the matplotlib interface.
"""

import numpy as np
from .. import _const


__all__ = ['MatplotlibPlotter']


class MatplotlibPlotter(object):
    '''
    Provides Methods to modify figures and axes objects for convenient
    plotting.
    It also autogenerates savenames and annotates the plot if a reader
    is given. A reader can be a dumpreader or a simulationreder.
    '''

    import matplotlib.ticker
    axesformatterx = matplotlib.ticker.ScalarFormatter()
    axesformatterx.set_powerlimits((-2, 3))
    axesformattery = matplotlib.ticker.ScalarFormatter()
    axesformattery.set_powerlimits((-2, 3))

    from matplotlib.colors import LinearSegmentedColormap
    efieldcdict = {'red': ((0, 0, 0), (1, 1, 1)),
                   'green': ((0, 0, 0), (1, 0, 0)),
                   'blue': ((0, 1, 1), (1, 0, 0)),
                   'alpha': ((0, 1, 1), (0.5, 0, 0), (1, 1, 1))}
    symmap = LinearSegmentedColormap('EField', efieldcdict, 1024)

    def __init__(self, reader, outdir='./', autosave=False, project=None):
        self.autosave = autosave
        self.reader = reader
        self.outdir = outdir
        self.project = project
        self._globalnameprefix = ''
        self._savenamesused = []

    def __len__(self):
        return len(self._savenamesused)

    @property
    def globalnameprefix(self):
        return self._globalnameprefix

    @globalnameprefix.setter
    def globalnameprefix(self, value):
        if value is None:
            value = ''
        if isinstance(value, str):
            self._globalnameprefix = value
        else:
            raise Exception('globalnameprefix has to be of string type'
                            ' (some other type found).')
        return

    def savename(self, key, ext='.png'):
        name = self.reader.name + \
            '_' + self.globalnameprefix + str(len(self._savenamesused)) + \
            '_' + key
        name = name.replace('/', '_').replace(' ', '')
        name = self.outdir + name
        nametmp = name + '_%d'
        i = 0
        while name in self._savenamesused:
            i = i + 1
            name = nametmp % i
        self._savenamesused.append(name)
        # print name
        return name + ext

    def lastsavename(self):
        '''
        returns the last savenme. If there wasnt a last a new savename
        is created.
        '''
        if len(self._savenamesused) == 0:
            return self._savename('lastsavename')
        else:
            return self._savenamesused[-1]
        return

    def savefig(self, fig, key, dpi=160, facecolor=(1, 1, 1, 0.01)):
        fig.set_size_inches(9, 7)
        savename = self.savename(key)
        fig.savefig(savename, dpi=dpi, facecolor=facecolor,
                    transparent=True)
        return

    @staticmethod
    def annotate(figorax, title=None, time=None, step=None, project=None, dump=None,
                 infostring=None, infos=None):
        if title:
            if hasattr(figorax, 'set_title'):  # figorax is ax
                figorax.set_title(title)
            else:  # figorax is fig
                figorax.suptitle(title)
        nexttext = ''
        if time:
            if isinstance(time, float):
                nexttext = '{:.1f} fs'.format(1e15 * time)
            else:
                nexttext = str(time)
        if step:
            if isinstance(step, (int, long, float)):
                nexttext += ', step: {:6.0f}'.format(step)
            else:
                nexttext += ' ' + str(step)
        if nexttext != '':
            figorax.text(0.92, 0.965, nexttext, horizontalalignment='right')
        if project:
            figorax.text(0.03, 0.965, project, horizontalalignment='left')
        if dump:
            figorax.text(0.03, 0.94, str(dump), horizontalalignment='left')
        if infostring:
            figorax.text(0.92, 0.93, infostring, ha='right')
        if infos is not None and infos != [] and infos != ['']:
            figorax.text(0.5, 0.87, str(infos), ha='center')
        return

    @staticmethod
    def annotate_fromfield(figorax, field):
        MatplotlibPlotter.annotate(figorax, title=field.label,
                                   infostring=field.infostring,
                                   infos=field.infos)
        return

    @staticmethod
    def annotate_fromreader(figorax, reader):
        try:
            MatplotlibPlotter.annotate(figorax,
                                       time=reader.time(),
                                       step=reader.timestep(),
                                       project=reader.name)
        except AttributeError:
            MatplotlibPlotter.annotate(figorax)
        return

    @staticmethod
    def symmetricclimaximage(aximage):
        """
        symmetrize the clim around 0.
        """
        bound = max(abs(np.asarray(aximage.get_clim())))
        aximage.set_clim(-bound, bound)
        return

    @staticmethod
    def symmetricclim(ax):
        """
        symmetrize the clim around 0.
        """
        MatplotlibPlotter.symmetricclimaximage(ax.images[0])
        return

    @staticmethod
    def addaxislabels(ax, field):
        if len(field.axes) > 0:
            ax.set_xlabel(field.axes[0].label)
        if len(field.axes) > 1:
            ax.set_ylabel(field.axes[1].label)
        return

    @staticmethod
    def addField1d(ax, field, log10plot=True,
                   xlim=None, ylim=None, scaletight=None):
        assert field.dimensions == 1, 'Field needs to be 1 dimensional'
        ax.plot(field.grid, field.matrix, label=field.label)
        ax.xaxis.set_major_formatter(MatplotlibPlotter.axesformatterx)
        ax.yaxis.set_major_formatter(MatplotlibPlotter.axesformattery)
        if log10plot and ((field.matrix < 0).sum() == 0) \
                and any(field.matrix > 0):
            ax.set_yscale('log')  # sets the axis to log scale AND overrides
            # our previously set axesformatter to the default
            # matplotlib.ticker.LogFormatterMathtext.
        MatplotlibPlotter.addaxislabels(ax, field)
        ax.autoscale(tight=scaletight)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        return ax

    @staticmethod
    def addFields1d(ax, *fields, **kwargs):
        # only write infos to Image if all infos of all fields are equal.
        clearinfos = not all([str(f.infos) == str(fields[0].infos)
                              for f in fields])
        infostrings = []
        for field in fields:
            if field.dimensions <= 0:
                continue
            infostrings.append(field.infostring)
            if clearinfos:
                field.infos = []
            MatplotlibPlotter.addField1d(ax, field, **kwargs)
        MatplotlibPlotter.annotate_fromfield(ax, field)
        MatplotlibPlotter.annotate(ax, infostring=str(infostrings))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        return

    # add lineouts to 2D-plots
    @staticmethod
    def _addxlineout(ax0, m, extent, log10=False):
        ax = ax0.twinx()
        l = m.mean(axis=0)
        if log10:
            l = np.log10(l)
        x = np.linspace(extent[0], extent[1], len(l))
        ax.plot(x, l, 'k', lw=1)
        ax.autoscale(tight=True)
        return ax

    @staticmethod
    def _addylineout(ax0, m, extent, log10=False):
        ax = ax0.twiny()
        # ax.set_xlim(ax.get_xlim()[::-1])
        l = m.mean(axis=1)
        if log10:
            l = np.log10(l)
        x = np.linspace(extent[2], extent[3], len(l))
        ax.plot(l, x, 'k', lw=1)
        ax.autoscale(tight=True)
        # ax.spines['top'].set_position(('axes',0.9))
        return ax

    @staticmethod
    def addField2d((fig, ax), field, log10plot=True, interpolation='none',
                   contourlevels=np.array([]), saveandclose=True, xlim=None,
                   ylim=None, clim=None,
                   savecsv=False, lineoutx=False, lineouty=False):
        assert field.dimensions == 2, 'Field needs to be 2 dimensional'
        ax.xaxis.set_major_formatter(MatplotlibPlotter.axesformatterx)
        ax.yaxis.set_major_formatter(MatplotlibPlotter.axesformattery)
        if log10plot and not any(field.matrix.flatten() < 0) and \
                any(field.matrix.flatten() > 0):
            if field.islinear() and True:
                ax.imshow(np.log10(field.matrix.T), origin='lower',
                          aspect='auto', extent=field.extent, cmap='jet',
                          interpolation=interpolation)
            else:
                print 'using pcolormesh, this is experimental.'
                x, y = field.grid()
                ax.pcolormesh(x, y, np.log10(field.matrix.T), cmap='jet')
            fig.colorbar(ax.images[0], format='%3.1f')
            if clim:
                ax.images[0].set_clim(clim)
        else:
            log10plot = False
            ax.imshow(field.matrix.T, aspect='auto', origin='lower',
                      extent=field.extent, cmap=MatplotlibPlotter.symmap,
                      interpolation=interpolation)
            if clim:
                ax.images[0].set_clim(clim)
            MatplotlibPlotter.symmetricclim(ax)
            fig.colorbar(ax.images[0], format='%6.0e')

        if contourlevels.size != 0:  # Draw contour lines
            ax.contour(field.matrix.T, contourlevels, hold='on',
                       extent=field.extent())
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if lineoutx:
            self._addxlineout(ax, field.matrix.T, field.extent,
                              log10=log10plot)
        if lineouty:
            self._addylineout(ax, field.matrix.T, field.extent,
                              log10=log10plot)
        MatplotlibPlotter.addaxislabels(ax, field)
        MatplotlibPlotter.annotate_fromfield(ax, field)
        return

    def _plotfinalize(self, fig):
        self.annotate_fromreader(fig, self.reader)
        return

    def plotFields1d(self, *fields, **kwargs):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        name = kwargs.pop('name') if 'name' in kwargs else fields[0].name
        MatplotlibPlotter.addFields1d(ax, *fields, **kwargs)
        self._plotfinalize(fig)
        if self.autosave:
            self.savefig(fig, name)
        if 'savecsv' in kwargs and kwargs['savecsv']:
            for field in fields:
                if field.dimensions == 0:
                    continue
                field.exporttocsv(self.lastsavename() + field.label + '.csv')
        return fig

    def plotField2d(self, field, name=None, **kwargs):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        MatplotlibPlotter.addField2d((fig, ax), field, **kwargs)
        self._plotfinalize(fig)
        if self.autosave:
            self.savefig(fig, name if name else field.name)
        return fig

    def plotField(self, field, autoreduce=True, maxlen=6000, name=None,
                  **kwargs):
        '''
        This is the main method, that should be used for plotting.
        '''
        if autoreduce:
            field.autoreduce(maxlen=maxlen)
        if field is None:
            ret = self._skipplot('none')
        elif field.dimensions <= 0:
            ret = self._skipplot(field.name)
        elif field.dimensions == 1:
            ret = self.plotFields1d(field, **kwargs)
        elif field.dimensions == 2:
            ret = self.plotField2d(field, **kwargs)
        else:
            raise Exception('3D not implemented')
        return ret

    def _skipplot(self, key):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.figtext(0.5, 0.5, 'No data available.', ha='center')
        print 'Skipped Plot: ' + self.lastsavename()
        return fig

    def plotFields(self, *fields, **kwargs):
        ret = [self.plotField(field, **kwargs) for field in fields]
        return ret

    def plotallderived(self, dumpreader):
        '''
        plots all fields dumped.
        '''
        from .. import analyzer
        try:
            derived = dumpreader.getderived()
        except AttributeError:
            return
        fa = analyzer.FieldAnalyzer(dumpreader)
        fields = fa.createfieldsfromkeys(*derived)
        for f in fields:
            self.plotField(f)
        return


