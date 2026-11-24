"""Scripted human intrusion for the Spatial bowl-to-plate scene, in metres."""
from xml.etree.ElementTree import SubElement

import numpy as np

from .level_three import _bounds
from .task_path import task_waypoints


def add_hand(model, cup_root):
    skin = '.78 .48 .34 1'
    sleeve = '.14 .31 .40 1'
    def body(name):
        body_name = 'human_'+name
        # The scripted grip owns the cup until release. Suppress internal
        # grip forces; the robot and the tabletop still collide with both.
        mug = model.worldbody.find(".//body[@name='"+cup_root+"']")
        for part in mug.iter('body'):
            SubElement(model.contact, 'exclude', body1=body_name, body2=part.get('name'))
        return SubElement(model.worldbody, 'body', name=body_name, mocap='true', pos='0 0 -6')
    def shape(parent, name, kind, size, color=skin, **attrs):
        SubElement(parent, 'geom', name='human_'+name, type=kind, size=size,
                   group='1', rgba=color, **attrs)
    palm = body('hand')
    shape(palm, 'sleeve', 'capsule', '.041', sleeve, fromto='.05 .19 .094 .065 .49 .136')
    shape(palm, 'forearm', 'capsule', '.030', fromto='.049 .110 .084 .051 .205 .096')
    shape(palm, 'palm', 'ellipsoid', '.042 .057 .019', pos='.045 .078 .084')
    for i, length in enumerate((.059,.070,.066,.049)):
        finger = body('digit'+str(i))
        x = .019 + .016*i
        y = .046 + abs(i-1)*.003
        z = .086-i*.0015
        radius=.0083-i*.0004
        coords = [(x,y,z), (x,y-length*.52,z+.002),
                  (x,y-length*.83,z-.012), (x-.003,y-length*.89,z-.027)]
        for j,(a,b) in enumerate(zip(coords,coords[1:])):
            shape(finger, f'finger{i}_{j}', 'capsule', str(radius),
                  fromto=' '.join(str(c) for c in a+b))
    thumb = body('thumb')
    shape(thumb, 'thumb_base', 'capsule', '.011', fromto='.016 .079 .079 -.001 .055 .068')
    shape(thumb, 'thumb_tip', 'capsule', '.009', fromto='-.001 .055 .068 -.003 .026 .070')


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t*t*(3.-2.*t)


class HandPlacesCup:
    """The hand carries a free mug by scripted pose; gravity takes over on release."""
    def __init__(self, domain, speed, delay):
        if not np.isfinite([speed, delay]).all() or speed <= 0 or delay < 0:
            raise ValueError('Hand preview requires positive speed and nonnegative delay')
        self.sim = sim = domain.sim
        m,d = sim.model._model, sim.data._data
        self.cup = domain.obj_body_id['red_coffee_mug_obstacle_1']
        joint = m.body_jntadr[self.cup]
        self.qa, self.va = m.jnt_qposadr[joint], m.jnt_dofadr[joint]
        self.mocaps = [m.body_mocapid[sim.model.body_name2id('human_'+name)] for name in
                       ['hand', 'digit0', 'digit1', 'digit2', 'digit3', 'thumb']]
        self.hand_geoms = []
        self.cup_geoms = []
        self.robot_geoms = set()
        self.human_geoms = set()
        for g in range(m.ngeom):
            name = sim.model.geom_id2name(g) or ''
            owner = m.geom_bodyid[g]
            if name.startswith('human_'):
                self.human_geoms.add(g)
                if name not in ('human_sleeve','human_forearm'):
                    self.hand_geoms.append(g)
            if name.startswith(('robot0_', 'gripper0_')):
                self.robot_geoms.add(g)
            while owner and owner != self.cup:
                owner=m.body_parentid[owner]
            if owner == self.cup and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.cup_geoms.append(g)
        if np.max(abs(d.xpos[self.cup,:2])) > .55:
            raise ValueError('Choose an initial state with an active red mug, e.g. Spatial 0 state 4')
        # State files start movable objects above the table. Settle the scene
        # before deriving waypoints and starting the event clock.
        d.mocap_pos[self.mocaps] = [0.,0.,-6.]
        d.qpos[self.qa:self.qa+3] = [0.,.78,1.]
        sim.forward()
        for _ in range(domain.control_freq):
            domain.step([0.]*6+[-1.])
        self.labels, self.path = task_waypoints(domain)
        # The library asset's free-joint origin is at the base, off the cup axis.
        d.qpos[self.qa:self.qa+7] = [0,0,1,1,0,0,0]
        sim.forward()
        bottom = min(_bounds(m,d,g)[0][2] for g in self.cup_geoms)-1
        table = sim.model.geom_name2id('table_collision')
        top = _bounds(m,d,table)[1][2]
        plate = self.path[-1]
        self.drop = np.array([plate[0]+.085,plate[1]-.106,top-bottom+.001])
        cross = .52*self.path[0]+.48*self.path[1]
        self.via = cross- [.045,.135,.085]
        self.entry = self.via.copy()
        self.entry[1] = .63
        self.above = self.drop + [0.,0.,.038]
        self.speed, self.delay = speed, delay
        self.times = np.r_[0., np.cumsum([np.linalg.norm(b-a)/speed for a,b in
                         zip([self.entry,self.via,self.above],[self.via,self.above,self.drop])])]
        self.release_s = delay+self.times[-1]+.20
        self.withdraw_s = self.release_s+.28
        self.duration = self.withdraw_s+np.linalg.norm(self.entry-self.drop)/speed+.50
        self.t0 = d.time
        self.released = self.touched_hand = self.touched_cup = False

    def contacts(self):
        d=self.sim.data._data
        for c in d.contact[:d.ncon]:
            a,b=int(c.geom1),int(c.geom2)
            if b in self.robot_geoms:
                a,b=b,a
            if a in self.robot_geoms and c.dist<=0.:
                self.touched_hand |= b in self.human_geoms
                self.touched_cup |= b in self.cup_geoms

    def apply(self, sim):
        d = sim.data._data
        self.contacts()
        t = d.time-self.t0
        u = max(0.,t-self.delay)
        points = [self.entry,self.via,self.above,self.drop]
        j = min(2,np.searchsorted(self.times[1:],u,side='right'))
        f = smooth((u-self.times[j])/(self.times[j+1]-self.times[j]))
        pos = (1-f)*points[j]+f*points[j+1]
        opening = smooth((t-self.release_s)/.28)
        self.phase = 'Carry in' if u < self.times[1] else 'Lower cup'
        if t < self.delay:
            self.phase = 'At table edge'
        if t < self.release_s or not self.released:
            d.qpos[self.qa:self.qa+7] = [*pos,1.,0.,0.,0.]
            d.qvel[self.va:self.va+6] = 0.
        if t >= self.release_s:
            self.released = True
            self.phase = 'Release'
        if t >= self.withdraw_s:
            w = smooth((t-self.withdraw_s)/(self.duration-.5-self.withdraw_s))
            pos = (1-w)*self.drop+w*self.entry + [0,0,.09*np.sin(np.pi*w)]
            self.phase = 'Withdraw' if w<1 else 'Cup remains'
        for k,i in enumerate(self.mocaps):
            pivot = np.zeros(3)
            axis = np.array([1.,0.,0.])
            angle = 0.
            if k in (1,2,3,4):
                n = k-1
                pivot = np.array([.019+.016*n, .046+abs(n-1)*.003, .086-n*.0015])
                angle = -.47*opening
            elif k==5:
                pivot = np.array([.016,.079,.079])
                axis = np.array([0.,0.,1.])
                angle = -.38*opening
            rot_pivot = (pivot*np.cos(angle) + np.cross(axis,pivot)*np.sin(angle)
                         + axis*np.dot(axis,pivot)*(1-np.cos(angle)))
            d.mocap_pos[i] = pos+pivot-rot_pivot
            d.mocap_quat[i] = np.r_[np.cos(angle/2),axis*np.sin(angle/2)]
        sim.forward()
