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
# Stephan Kuschel 2017

from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
import collections
import numexpr as ne

__all__ = ['ScalarProperty']


class ScalarProperty(object):

    _evaluate = staticmethod(ne.evaluate)

    def __init__(self, expr, name=None, unit=None, symbol=None):
        '''
        Represents a scalar particle property.

        expr - The expression how to calcualte the value (string).
        name - the name the property can be accessed by.
        unit - unit of property.
        symbol - symbol used in formulas. Defaults to 'name' if omitted.
        '''
        self._expr = expr
        self._name = name
        self._unit = unit
        self._symbol = symbol
        self._func_cache = None  # Optimized numexpr function if available

    @property
    def name(self):
        return self._name

    @property
    def expr(self):
        return self._expr

    @property
    def unit(self):
        return self._unit

    @property
    def symbol(self):
        return self.name if self._symbol is None else self._symbol

    @property
    def _func(self):
        '''
        The optimized numexpr function.
        '''
        if self._func_cache is None:
            self._func_cache = ne.NumExpr(self.expr)
        return self._func_cache

    @property
    def input_names(self):
        '''
        The list of variables used within this expression.
        '''
        return self._func.input_names

    def evaluate(self, vars):
        '''
        vars must be a dictionary containing variables used
        within the expression "expr".
        '''
        args = [vars[v] for v in self.input_names]
        return self._func(*args)

    def __iter__(self):
        for k in ['name', 'expr', 'unit', 'symbol']:
            yield k, getattr(self, k)

    def __str__(self):
        return self.expr

    def __repr__(self):
        formatstring = 'ScalarProperty("{expr}", name="{name}", unit="{unit}", symbol="{symbol}")'
        return formatstring.format(**dict(self))


class ScalarPropertyContext(collections.Mapping):

    def __init__(self):
        '''
        only used internally to store the list of known particle
        properties (scalar particle values). Provides functions to add
        or remove properties during runtime.
        '''
        self._mapping = dict()

    def __getitem__(self, key):
        return self._mapping[key]

    def __iter__(self):
        for k in self._mapping:
            yield k

    def __len__(self):
        return len(self._mapping)

    def add(self, sp):
        '''
        register a new ScalarProperty, such, that
        this will be identified by its symbol.
        '''
        if sp.symbol in self._mapping:
            raise KeyError('The symbol {} is already known!'.format(sp.symbol))
        if sp.symbol is None:
            raise ValueError('Impossible to add the anonymous ScalarProperty {}'.format(str(sp)))
        self._mapping.update({sp.symbol: sp})

    def __repr__(self):
        # order alphabetically to increase readability
        scalars = sorted(list(self))
        s = [str(k) + ' = ' + self[k].expr for k in scalars]
        s.append('--> {} known particle scalars.'.format(len(s)))
        return '\n'.join(s)

    def remove(self, symbol):
        self._mapping.pop(symbol)

    def __call__(self, expr):
        '''
        tries to identify the ScalarProperty by its expression or symbol.
        '''
        if expr in self:
            return self[expr]
        for k in self:
            if self[k].expr == expr:
                return self[k]
        # if not found return an anonymous ScalarProperty
        return ScalarProperty(expr)


# template: ScalarProperty('', '', ''),
_defaultscalars = [
    ScalarProperty('time', 'time', 's'),
    ScalarProperty('time', 'time', 's', symbol='t'),
    ScalarProperty('weight', 'weight', 'npartpermacro'),
    ScalarProperty('weight', 'weight', 'npartpermacro', symbol='w'),
    ScalarProperty('id', 'id', ''),
    ScalarProperty('mass', 'mass', 'kg'),
    ScalarProperty('mass', 'mass', 'kg', symbol='m'),
    ScalarProperty('mass / atomic_mass', 'mass_u', 'amu'),
    ScalarProperty('mass / atomic_mass', 'mass_u', 'amu', symbol='m_u'),
    ScalarProperty('charge', 'charge', 'C'),
    ScalarProperty('charge', 'charge', 'C', symbol='q'),
    ScalarProperty('charge / elementary_charge', 'charge_e', ''),
    ScalarProperty('charge / elementary_charge', 'charge_e', '', symbol='q_e'),
    ScalarProperty('x', 'x', 'm'),
    ScalarProperty('x * 1e6', 'x_um', r'$\mu$ m'),
    ScalarProperty('y', 'y', 'm'),
    ScalarProperty('y * 1e6', 'y_um', r'$\mu$ m'),
    ScalarProperty('z', 'z', 'm'),
    ScalarProperty('y * 1e6', 'z_um', r'$\mu$ m'),
    ScalarProperty('px', 'px', 'kg*m/s'),
    ScalarProperty('py', 'py', 'kg*m/s'),
    ScalarProperty('pz', 'pz', 'kg*m/s'),
    ScalarProperty('sqrt(px**2 + py**2 + pz**2)', 'p', 'kg*m/s'),
    ScalarProperty('(px**2 + py**2 + pz**2)/(mass * c)**2', '_np2', ''),
    ScalarProperty('_np2 / (sqrt(1 + _np2) + 1)', 'gamma_m1', ''),
    ScalarProperty('_np2 / (sqrt(1 + _np2) + 1) + 1', 'gamma', ''),
    ScalarProperty('gamma * mass', 'gamma_m', 'kg'),
    ScalarProperty('beta * c', 'v', 'm/s'),
    ScalarProperty('px / (gamma * mass)', 'vx', 'm/s'),
    ScalarProperty('py / (gamma * mass)', 'vy', 'm/s'),
    ScalarProperty('pz / (gamma * mass)', 'vz', 'm/s'),
    ScalarProperty('sqrt(gamma**2 - 1) / gamma', 'beta', ''),
    ScalarProperty('vx / c', 'betax', ''),
    ScalarProperty('vy / c', 'betay', ''),
    ScalarProperty('vz / c', 'betaz', ''),
    ScalarProperty('mass * c**2', 'Eruhe', 'J'),
    ScalarProperty('gamma_m1 * mass * c**2', 'Ekin', 'J'),
    ScalarProperty('Ekin / elementary_charge / 1e6', 'Ekin_MeV', 'MeV'),
    ScalarProperty('Ekin / elementary_charge / 1e6 / mass_u', 'Ekin_MeV_amu', 'MeV/u'),
    ScalarProperty('Ekin / elementary_charge / 1e6 / mass_u * charge_e',
                   'Ekin_MeV_qm', 'MeV * q/m'),
    ScalarProperty('Ekin / elementary_charge / 1e3', 'Ekin_keV', 'keV/u'),
    ScalarProperty('Ekin / elementary_charge / 1e3 / mass_u', 'Ekin_keV_amu', 'keV/u'),
    ScalarProperty('Ekin / elementary_charge / 1e3 / mass_u * charge_e',
                   'Ekin_keV_qm', 'keV * q/m'),
    ScalarProperty('arctan2(py, px)', 'angle_xy', 'rad'),
    ScalarProperty('arctan2(px, py)', 'angle_yx', 'rad'),
    ScalarProperty('arctan2(pz, py)', 'angle_yz', 'rad'),
    ScalarProperty('arctan2(py, pz)', 'angle_zy', 'rad'),
    ScalarProperty('arctan2(px, pz)', 'angle_zx', 'rad'),
    ScalarProperty('arctan2(pz, px)', 'angle_xz', 'rad'),
    ScalarProperty('arctan2(sqrt(py**2 + pz**2), px)', 'angle_xaxis', 'rad'),
    ScalarProperty('arctan2(sqrt(pz**2 + px**2), py)', 'angle_yaxis', 'rad'),
    ScalarProperty('arctan2(sqrt(px**2 + py**2), pz)', 'angle_zaxis', 'rad'),
    ScalarProperty('sqrt(x**2 + y**2)', 'r_xy', 'm'),
    ScalarProperty('sqrt(y**2 + z**2)', 'r_yz', 'm'),
    ScalarProperty('sqrt(z**2 + x**2)', 'r_zx', 'm'),
    ScalarProperty('sqrt(x**2 + y**2 + z**2)', 'r_xyz', 'm')
    ]


# ---- List of default scalar particle properties and how to calculate them
def createdefaultscalarcontext():
    ret = ScalarPropertyContext()
    for _s in _defaultscalars:
        ret.add(_s)
    return ret
