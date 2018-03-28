#
# This file is part of postpic.
#
# postpic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# postpic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with postpic. If not, see <http://www.gnu.org/licenses/>.
#
# Stephan Kuschel 2014-2017
# Alexander Blinne, 2017
"""
This package provides the MatplotlibPlotter Class.

This Class can be used to plot Field Objects using the matplotlib interface.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import warnings


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
    efieldcdict = {'red': ((0, 0, 0),
                           (0.5, 1, 1),
                           (1.0, 1, 1)),
                   'green': ((0, 0, 0),
                             (0.5, 1, 1),
                             (1, 0, 0)),
                   'blue': ((0, 1, 1),
                            (0.5, 1, 1),
                            (1, 0, 0))}
    symmap = LinearSegmentedColormap('EField', efieldcdict, 1024)

    def __init__(self, reader, outdir='./', autosave=False, project=None,
                 ext='png', size_inches=(9, 7), dpi=160, facecolor=(1, 1, 1, 0.01),
                 transparent=False):
        self._ext = ext
        self.autosave = autosave
        self.reader = reader
        self.outdir = outdir
        self._project = project
        self.size_inches = size_inches
        self.dpi = dpi
        self.facecolor = facecolor
        self.transparent = transparent
        self._savenamesused = []

    def __len__(self):
        return len(self._savenamesused)

    @property
    def project(self):
        return self._project if self._project else ''

    def savename(self, key, ext=None):
        if not ext:
            ext = self._ext
        name = self.project + '_' + self.reader.name + \
            '_' + str(len(self._savenamesused)) + '_' + key
        name = name.replace('/', '_').replace(' ', '')
        name = self.outdir + name
        nametmp = name + '_%d'
        i = 0
        while name in self._savenamesused:
            i = i + 1
            name = nametmp % i
        self._savenamesused.append(name)
        # print name
        return name + '.' + ext

    def lastsavename(self):
        '''
        returns the last savenme. If there wasnt a last a new savename
        is created.
        '''
        if len(self._savenamesused) == 0:
            return self.savename('lastsavename')
        else:
            return self._savenamesused[-1]
        return

    def savefig(self, fig, key):
        savename = self.savename(key)
        fig.savefig(savename, dpi=self.dpi, facecolor=self.facecolor,
                    transparent=self.transparent)
        return

    @staticmethod
    def settext_fig(fig, title=None, ur=None, ur2=None, ul=None, ul2=None,
                    center=None):
        if title:
            fig.suptitle(title)
        if ur:
            fig.text(0.92, 0.965, ur, ha='right')
        if ur2:
            fig.text(0.92, 0.93, ur2, ha='right')
        if ul:
            fig.text(0.03, 0.965, ul, horizontalalignment='left')
        if ul2:
            fig.text(0.03, 0.93, ul2, horizontalalignment='left')
        if center:
            fig.text(0.5, 0.87, center, horizontalalignment='center')

    @staticmethod
    def settext_ax(ax, title=None, ur=None, ur2=None, ul=None, ul2=None,
                   center=None):
        if title:
            ax.set_title(title)
        if ur2:
            ax.text(0.99, 1.01, ur2, ha='right', transform=ax.transAxes)
        if ur:
            ax.text(0.99, 0.97, ur, ha='right', transform=ax.transAxes)
        if ul2:
            ax.text(0.01, 1.01, ul2, horizontalalignment='left', transform=ax.transAxes)
        if ul:
            ax.text(0.01, 0.97, ul, horizontalalignment='left', transform=ax.transAxes)
        if center:
            ax.text(0.5, 0.87, center, horizontalalignment='center', transform=ax.transAxes)

    @staticmethod
    def annotate(figorax, title=None, time=None, step=None, project=None, dump=None,
                 infostring=None, infos=None):
        ur = ''
        if time:
            if isinstance(time, float):
                ur = '{:.1f} fs'.format(1e15 * time)
            else:
                ur = str(time)
        if step:
            if isinstance(step, (int, float)):
                ur += ', step: {:6.0f}'.format(step)
            else:
                ur += ' ' + str(step)
        ul = project
        ul2 = dump
        ur2 = infostring
        center = None if infos == [] or infos == [''] else infos
        import matplotlib
        func = MatplotlibPlotter.settext_ax if isinstance(figorax, matplotlib.axes.Axes) \
            else MatplotlibPlotter.settext_fig
        func(figorax, title, ur, ur2, ul, ul2, center)
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
                                       dump=reader.name)
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
        field = field.squeeze()
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
        lout = m.mean(axis=0)
        if log10:
            lout = np.log10(lout)
        x = np.linspace(extent[0], extent[1], len(lout))
        ax.plot(x, lout, 'k', lw=1)
        ax.autoscale(tight=True)
        return ax

    @staticmethod
    def _addylineout(ax0, m, extent, log10=False):
        ax = ax0.twiny()
        # ax.set_xlim(ax.get_xlim()[::-1])
        lout = m.mean(axis=1)
        if log10:
            lout = np.log10(lout)
        x = np.linspace(extent[2], extent[3], len(lout))
        ax.plot(lout, x, 'k', lw=1)
        ax.autoscale(tight=True)
        # ax.spines['top'].set_position(('axes',0.9))
        return ax

    @staticmethod
    def addField2d(figax, field, log10plot=True, interpolation='none',
                   contourlevels=np.array([]), saveandclose=True, xlim=None,
                   ylim=None, clim=None,
                   savecsv=False, lineoutx=False, lineouty=False, **kwargs):
        field = field.squeeze()
        (fig, ax) = figax
        assert field.dimensions == 2 or (field.dimensions == 3 and field.shape[2] in [3, 4]), \
            'Field needs to be 2 dimensional'

        color_image = field.dimensions == 3
        maximum = None
        if color_image:
            maximum = np.max(field.matrix)
            field = field/maximum

        ax.xaxis.set_major_formatter(MatplotlibPlotter.axesformatterx)
        ax.yaxis.set_major_formatter(MatplotlibPlotter.axesformattery)
        if log10plot and not any(field.matrix.flatten() < 0) and \
                any(field.matrix.flatten() > 0) and not color_image:
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'jet'
            if 'aspect' not in kwargs:
                kwargs['aspect'] = 'auto'
            if all(field.islinear()):
                im = ax.imshow(np.log10(field.matrix.T), origin='lower',
                               extent=field.extent,
                               interpolation=interpolation, **kwargs)
            elif not color_image:
                x, y = [ax.grid_node for ax in field.axes]
                if 'aspect' in kwargs:
                    del kwargs['aspect']
                im = ax.pcolormesh(x, y, np.log10(field.matrix.T), **kwargs)
            else:
                raise ValueError("color images with non-linear axes not supported by this "
                                 "function.")
            fig.colorbar(im, format='%3.1f')
            if clim:
                im.set_clim(clim)
        else:
            log10plot = False
            if 'cmap' not in kwargs:
                kwargs['cmap'] = MatplotlibPlotter.symmap
            if 'aspect' not in kwargs:
                kwargs['aspect'] = 'auto'
            if all(field.islinear()):
                im = ax.imshow(np.swapaxes(field.matrix, 0, 1), origin='lower',
                               extent=field.extent[:4], interpolation=interpolation, **kwargs)
            elif not color_image:
                x, y = [ax.grid_node for ax in field.axes]
                if 'aspect' in kwargs:
                    del kwargs['aspect']
                im = ax.pcolormesh(x, y, field.matrix.T, **kwargs)
            else:
                raise ValueError("color images with non-linear axes not supported by this "
                                 "function.")
            if clim:
                im.set_clim(clim)
            else:
                MatplotlibPlotter.symmetricclim(ax)
            if not color_image:
                fig.colorbar(im, format='%6.0e')

        if contourlevels.size != 0:  # Draw contour lines
            ax.contour(field.matrix.T, contourlevels, hold='on',
                       extent=field.extent)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if lineoutx:
            MatplotlibPlotter._addxlineout(ax, field.matrix.T, field.extent,
                                           log10=log10plot)
        if lineouty:
            MatplotlibPlotter._addylineout(ax, field.matrix.T, field.extent,
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
        name = kwargs.pop('name', fields[0].name)
        MatplotlibPlotter.addFields1d(ax, *fields, **kwargs)
        self._plotfinalize(fig)
        self.annotate(fig, project=self.project)
        self.annotate(ax, infostring=str([f.infostring for f in fields]))
        fig.set_size_inches(*self.size_inches)
        if self.autosave:
            self.savefig(fig, name)
            plt.close(fig)
            fig = None
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
        self.annotate(fig, project=self.project)
        self.annotate(ax, infostring=field.infostring)
        fig.set_size_inches(*self.size_inches)
        if self.autosave:
            self.savefig(fig, name if name else field.name)
            plt.close(fig)
            fig = None
        return fig

    def plotField(self, field, autoreduce=True, maxlen=6000, name=None,
                  **kwargs):
        '''
        This is the main method, that should be used for plotting.
        '''
        field = field.squeeze()
        if autoreduce:
            field.autoreduce(maxlen=maxlen)
        if field is None:
            ret = self._skipplot('none')
        elif field.dimensions <= 0:
            ret = self._skipplot(name if name else field.name)
        elif field.dimensions == 1:
            if name:
                kwargs.update({'name': name})
            ret = self.plotFields1d(field, **kwargs)
        elif field.dimensions == 2 or (field.dimensions == 3 and field.shape[2] in [3, 4]):
            ret = self.plotField2d(field, name,  **kwargs)
        else:
            raise Exception('3D not implemented')
        return ret

    def _skipplot(self, key):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.figtext(0.5, 0.5, 'No data available.', ha='center')
        fig.set_size_inches(*self.size_inches)
        if self.autosave:
            self.savefig(fig, key)
            plt.close(fig)
            fig = None
        print('Skipped Plot.')
        return fig

    def plotFields(self, *fields, **kwargs):
        ret = [self.plotField(field, **kwargs) for field in fields]
        return ret

    def plotallderived(self, dumpreader):
        '''
        plots all fields dumped.
        '''
        try:
            derived = dumpreader.getderived()
        except AttributeError:
            return
        fields = dumpreader.createfieldsfromkeys(*derived)
        for f in fields:
            self.plotField(f)
        return
