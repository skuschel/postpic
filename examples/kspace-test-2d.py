#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys


def download(url, file):
    import urllib3
    import shutil
    import os
    if os.path.isfile(file):
        return True
    try:
        urllib3.disable_warnings()
        http = urllib3.PoolManager()
        print('downloading {:} ...'.format(file))
        with http.request('GET', url, preload_content=False) as r, open(file, 'wb') as out_file:
            shutil.copyfileobj(r, out_file)
        success = True
    except urllib3.exceptions.MaxRetryError:
        success = False
    return success

def main():
    import os
    import os.path as osp
    datadir = 'examples/kspace-test-2d'
    tarball = osp.join(datadir, 'kspace-test-2d-resources.tar.gz')
    tarball_url = 'https://blinne.net/files/postpic/kspace-test-2d-resources.tar.gz'
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    s = download(tarball_url, tarball)

    if s:
        import hashlib
        chksum = hashlib.sha256(open(tarball, 'rb').read()).hexdigest()
        if chksum != "45549cc85bf1f8c60840a870c7279ddbc5f14f2bb9ff5de2728586b6198a8e20":
            os.remove(tarball)
            s = False

    if not s:
        print('Failed to Download example data. Skipping this example.')
        return

    import tarfile
    tar = tarfile.open(tarball)
    tar.extractall(datadir)


    import matplotlib
    matplotlib.use('Agg')

    font = {'size'   : 12}
    matplotlib.rc('font', **font)

    import copy
    import postpic as pp
    import numpy as np
    import pickle

    try:
        import sdf
        sdfavail = True
    except ImportError:
        sdfavail = False

    # basic constants
    micro = 1e-6
    femto = 1e-15
    c = pp.PhysicalConstants.c

    # known parameters from simulation
    lam = 0.5 * micro
    k0 = 2*np.pi/lam
    f0 = pp.PhysicalConstants.c/lam

    if sdfavail:
        pp.chooseCode("EPOCH")

        dump = pp.readDump(osp.join(datadir, '0002.sdf'))
        plotter = pp.plotting.plottercls(dump, autosave=False)
        fields = dict()
        for fc in ['Ey', 'Bz']:
            fields[fc] = getattr(dump, fc)()
            fields[fc].saveto('0002_'+fc, compressed=False)
        t = dump.time()
        dt = t/dump.timestep()
        pickle.dump(dict(t=t, dt=dt), open(osp.join(datadir,'0002_meta.pickle'), 'wb'))
    else:
        fields = dict()
        for fc in ['Ey', 'Bz']:
            fields[fc] = pp.Field.loadfrom(osp.join(datadir,'0002_{}.npz').format(fc))
        meta = pickle.load(open(osp.join(datadir,'0002_meta.pickle'), 'rb'))
        t = meta['t']
        dt = meta['dt']
        plotter = pp.plotting.plottercls(None, autosave=False)

    Ey = fields['Ey']
    # grid spacing from field
    dx = [ax.grid[1] - ax.grid[0] for ax in Ey.axes]


    #w0 = 2*np.pi*f0
    #w0 = pp.PhysicalConstants.c * k0

    wn = np.pi/dt
    omega_yee = pp.helper.omega_yee_factory(dx=dx, dt=dt)

    wyee = omega_yee([k0,0])
    w0 = pp.helper.omega_free([k0,0])

    print('dx', dx)
    print('t', t)
    print('dt', dt)
    print('wn', wn)
    print('w0', w0)
    print('wyee', wyee)
    print('wyee/w0', wyee/w0)
    print('wyee/wn', wyee/wn)
    print('lam/dx[0]', lam/dx[0])

    print('cos(1/2 wyee dt)', np.cos(1/2 * wyee * dt))

    vg_yee    = c*np.cos(k0*dx[0]/2.0)/np.sqrt(1-(c*dt/dx[0]*np.sin(k0*dx[0]/2.0))**2)
    print('vg/c', vg_yee/c)

    r = np.sqrt(1.0 - (pp.PhysicalConstants.c * dt)**2 * (1/dx[0]*np.sin(1/2.0*k0*dx[0]))**2)
    print('r', r)


    # In[2]:



    omega_yee = pp.helper.omega_yee_factory(dx=Ey.spacing, dt=dt)
    lin_int_response_omega = pp.helper._linear_interpolation_frequency_response(dt)
    lin_int_response_k = pp.helper._linear_interpolation_frequency_response_on_k(lin_int_response_omega,
                                                                                Ey.fft().axes, omega_yee)

    lin_int_response_k_vac = pp.helper._linear_interpolation_frequency_response_on_k(lin_int_response_omega,
                                                                                Ey.fft().axes, pp.helper.omega_free)


    # In[3]:


    _=plotter.plotField(Ey[:,0.0])


    # In[4]:


    kspace = dict()
    component = "Ey"


    # In[5]:


    #key = 'default epoch map=yee, omega=vac'
    key = 'linresponse map=yee, omega=vac'
    if sdfavail:
        kspace[key] = abs(dump.kspace_Ey(solver='yee'))
    else:
        kspace[key] = abs(pp.helper.kspace(component, fields=dict(Ey=fields['Ey'],
                                                                Bz=fields['Bz'].fft()/lin_int_response_k),
                                interpolation='fourier'))
        #  using the helper function `kspace_epoch_like` would yield same result:
        # kspace[key] = abs(pp.helper.kspace_epoch_like(component, fields=fields, dt=dt, omega_func=omega_yee))
    normalisation = 1.0/np.max(kspace[key].matrix)
    kspace[key] *= normalisation
    kspace[key].name = r'corrected $\vec{k}$-space, $\omega_0=c|\vec{k}|$'
    kspace[key].unit = ''


    # In[6]:


    key = 'simple fft'
    kspace[key] = abs(Ey.fft()) * normalisation
    kspace[key].name = r'plain fft'
    kspace[key].unit = ''


    # In[7]:


    key = 'fourier'
    kspace[key] = abs(pp.helper.kspace(component,
                                    fields=fields,
                                    interpolation='fourier')
                    ) * normalisation
    kspace[key].name = r'naïve $\vec{k}$-space, $\omega_0=c|\vec{k}|$'
    kspace[key].unit = ''


    # In[8]:


    key = 'fourier yee'
    kspace[key] = abs(pp.helper.kspace(component,
                                    fields=fields, interpolation='fourier',
                                    omega_func=omega_yee)
                    ) * normalisation
    kspace[key].name = r'naïve $\vec{k}$-space, $\omega_0=\omega_\mathrm{grid}$'
    kspace[key].unit = ''


    # In[9]:


    key = 'linresponse map=yee, omega=yee'
    kspace[key] = abs(pp.helper.kspace(component, fields=dict(Ey=fields['Ey'],
                                                        Bz=fields['Bz'].fft()/lin_int_response_k),
                                interpolation='fourier', omega_func=omega_yee)
                    ) * normalisation
    kspace[key].name = r'corrected $\vec{k}$-space, $\omega_0=\omega_\mathrm{grid}$'
    kspace[key].unit = ''


    # In[10]:


    slices = [slice(360-120, 360+120), slice(120, 121)]


    # In[11]:


    keys = ['simple fft',
            'fourier yee',
            'fourier',
            'linresponse map=yee, omega=yee',
            'linresponse map=yee, omega=vac'
            ]
    figure2 = plotter.plotFields1d(*[kspace[k][slices] for k in keys],
                                log10plot=True, ylim=(5e-17, 5))
    figure2.set_figwidth(8)
    figure2.set_figheight(6)
    while figure2.axes[0].texts:
        figure2.axes[0].texts[-1].remove()

    figure2.axes[0].set_title('')
    figure2.axes[0].set_ylabel(r'$|E_y(k_x,0,0)|\, [a. u.]$')
    figure2.axes[0].set_xlabel(r'$k_x\,[m^{-1}]$')
    figure2.tight_layout()
    figure2.savefig(osp.join(datadir, 'gaussian-kspace.pdf'))

    print("Integrated ghost peaks")
    for k in keys:
        I = kspace[k][:0.0,:].integrate().matrix
        print(k, I)
        if k == 'linresponse map=yee, omega=vac':
            if I < 30000000.:
                print('linresponse map=yee, omega=vac value is low enough: YES' )
            else:
                print('linresponse map=yee, omega=vac value is low enough: NO' )
                print('Something is WRONG' )
                sys.exit(1)



    # In[13]:


    if sdfavail:
        kspace_ey = dump.kspace_Ey(solver='yee')
    else:
        kspace_ey = pp.helper.kspace_epoch_like(component, fields=fields, dt=dt, omega_func=omega_yee)
    complex_ey = kspace_ey.fft()
    envelope_ey_2d = abs(complex_ey)[:,:].squeeze()
    try:
        from skimage.restoration import unwrap_phase
        phase_ey = complex_ey.replace_data( unwrap_phase(np.angle(complex_ey)) )
    except ImportError:
        phase_ey = complex_ey.replace_data( np.angle(complex_ey) )
    phase_ey_2d = phase_ey[:,:].squeeze()


    # In[14]:


    ey = complex_ey.real[-0.1e-5:0.2e-5, :]
    #ey = Ey

    ey.name = r'$E_y$'
    ey.unit = r'$\frac{\mathrm{V}}{\mathrm{m}}$'

    figure = plotter.plotField2d(ey)
    figure.set_figwidth(6)
    figure.axes[0].set_title(r'')#$\Re E_y\ [\frac{\mathrm{V}}{\mathrm{m}}]$')
    figure.axes[0].set_xlabel(r'$x\,[µm]$')
    figure.axes[0].set_ylabel(r'$y\,[µm]$')

    import matplotlib.ticker as ticker
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e-6))
    figure.axes[0].xaxis.set_major_formatter(ticks_x)
    figure.axes[0].yaxis.set_major_formatter(ticks_x)

    figure.axes[0].images[0].colorbar.remove()
    figure.colorbar(figure.axes[0].images[0], format='%6.0e', pad=0.15, label=r'$\Re E_y\ [\frac{\mathrm{V}}{\mathrm{m}}]$')

    axes2 = figure.axes[0].twinx()
    axes2.set_ylabel(r'$\operatorname{Arg}(E_y)\, [\pi],\;|E_y|\, [a. u.]$')

    env = abs(complex_ey[-0.1e-5:0.2e-5,0.0])
    m = np.max(env)
    env = (env/m*40/np.pi).squeeze()
    p = phase_ey[-0.1e-5:0.2e-5,0.0].squeeze()/np.pi

    _ = axes2.plot(env.grid, env.matrix, label=r'$|E_y|\, [a. u.]$')
    _ = axes2.plot(p.grid, p.matrix, label=r'$\operatorname{Arg}(E_y)\, [\pi]$')

    handles, labels = axes2.get_legend_handles_labels()
    axes2.legend(handles, labels)

    #figure.axes[0].set_title(r'$E_y\,[V/m]$')

    figure.set_figwidth(6)
    figure.set_figheight(6)
    figure.tight_layout()
    figure.savefig(osp.join(datadir, 'gaussian-env-arg.pdf'))


    # In[ ]:


if __name__=='__main__':
    main()
