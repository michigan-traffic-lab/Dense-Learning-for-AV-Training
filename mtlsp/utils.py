import uuid
import numpy as np
import math
import torch
import torch.nn as nn
import bisect
import xml.etree.ElementTree as ET
import json, ujson
import os
if os.environ["mode"] == "testing":
    import conf.conf_testing as conf
elif os.environ["mode"] == "training":
    import conf.conf_training as conf
else:
    raise ValueError("Please set the mode to testing or training")


def generate_unique_bv_id():
    """Randomly generate an ID of the background vehicle

    Returns:
        str: ID of the background vehicle
    """
    return 'BV_'+str(uuid.uuid4())

def remap(v, x, y): 
    """Remap a value from one range to another.

    Args:
        v (float): Value to be remapped.
        x (list): Original range.
        y (list): Target range.

    Returns:
        float: Remapped value.
    """
    return y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])

def check_equal(x, y, error):
    """Check if x is approximately equal to y considering the given error.

    Args:
        x (float): Parameter 1.
        y (float): Parameter 2.
        error (float): Specified error.

    Returns:
        bool: True is x and y are close enough. Otherwise, False.
    """
    if abs(x-y) <= error:
        return True
    else:
        return False

def cal_dis_with_start_end_speed(v_start, v_end, acc, time_interval=1.0, v_low=20, v_high=40):
    """Calculate the travel distance with start and end speed and acceleration.

    Args:
        v_start (float): Start speed [m/s].
        v_end (float): End speed [m/s].
        acc (float): Acceleration [m/s^2].
        time_interval (float, optional): Time interval [s]. Defaults to 1.0.

    Returns:
        float: Travel distance in the time interval.
    """
    if v_end == v_low or v_end == v_high:
        t_1 = (v_end-v_start)/acc if acc != 0 else 0
        t_2 = time_interval - t_1
        dis = v_start*t_1 + 0.5*(acc)*(t_1**2) + v_end*t_2
    else:
        dis = ((v_start+v_end)/2)*time_interval
    return dis

def cal_euclidean_dist(veh1_position=None, veh2_position=None):
    """Calculate Euclidean distance between two vehicles.

    Args:
        veh1_position (tuple, optional): Position of Vehicle 1 [m]. Defaults to None.
        veh2_position (tuple, optional): Position of Vehicle 2 [m]. Defaults to None.

    Raises:
        ValueError: If the position of fewer than two vehicles are provided, raise error.

    Returns:
        float: Euclidean distance between two vehicles [m].
    """    
    if veh1_position is None or veh2_position is None:
        raise ValueError("Fewer than two vehicles are provided!")
    veh1_x, veh1_y = veh1_position[0], veh1_position[1]
    veh2_x, veh2_y = veh2_position[0], veh2_position[1]
    return math.sqrt(pow(veh1_x-veh2_x, 2)+pow(veh1_y-veh2_y, 2))

def load_trajs_from_fcdfile(file_path):
    """Load vehicle trajectories from the fcd file.

    Args:
        file_path (str): Path of the fcd file.

    Returns:
        dict: Vehicle trajectories.
    """
    tree = ET.parse(file_path)
    return fcd2dict(tree)

def fcd2dict(tree):
    """Convert fcd file to dictionary.

    Args:
        tree (xml.etree.ElementTree.ElementTree): Element tree of the fcd file.

    Returns:
        dict: Vehicle trajectories.
    """
    vehicle_trajs = {}
    root = tree.getroot()
    for time_step in root.getchildren():
        time = time_step.attrib["time"]
        vehicles_dict = {}
        for veh in time_step:
            veh_info = veh.attrib
            veh_id = veh_info["id"]
            vehicles_dict[veh_id] = veh_info
        vehicle_trajs[str(round(float(time),1))] = vehicles_dict
    return vehicle_trajs

def load_trajs_from_jsonfile(file_path_info):
    """Load vehicle trajectories from the json file.

    Args:
        file_path_info (tuple): Path of the json file.
        
    Returns:
        dict: Vehicle trajectories.
    """
    file_path = file_path_info[1]+"/"+str(file_path_info[0][1])+".fcd.json"
    with open(file_path) as fo:
        line = get_line(fo, file_path_info[0][2])
    fcd_info = ujson.loads(line)
    return json2dict(fcd_info)

def json2dict(json_info):
    """Convert json file to dictionary.

    Args:
        json_info (dict): Json file content.

    Returns:
        dict: Vehicle trajectories.
    """
    vehicle_trajs = {}
    for time_step_info in json_info["fcd-export"]["timestep"]:
        time = time_step_info["@time"]
        vehicles_dict = {}
        for veh in time_step_info["vehicle"]:
            veh_info = {}
            for key in veh:
                veh_info[key.split("@")[-1]] = veh[key]
            veh_id = veh_info["id"]
            vehicles_dict[veh_id] = veh_info
        vehicle_trajs[str(round(float(time),1))] = vehicles_dict
    return vehicle_trajs
        
def get_line(fp, line_number):
    """Get the line content based on the line number.

    Args:
        fp (file): File pointer.
        line_number (int): Line number.

    Returns:
        str: Line content.
    """
    for i, x in enumerate(fp):
        if i == line_number:
            return x
    return None

def update_vehicle_real_states(original_states, action, parameters, duration):
    """Get the next vehicle states based on simple vehicle dynamic model (bicycle model).

    Args:
        original_states (list): Vehicle states including longitudinal speed in vehicle coordinate, longitudinal position in road coordinate, lateral position in road coordinate, heading in absolute coordinate, lateral speed in vehicle coordinate, yaw rate in absolute coordinate.
        action (dict): Next action including longitudinal acceleration in vehicle coordinate and steering angle.
        parameters (dict): Vehicle dynamics parameters including a, L, m, Iz, Caf, Car.
        duration (float): Simulation time step.

    Returns:
        list: New vehicle states with the same format as the original states. 
    """    
    # first assume straight road
    au = action["acceleration"] # longitudinal acceleration in vehicle coordinate 
    deltaf = action["steering_angle"]/180*math.pi # steering angle in vehicle coordinate
    u = original_states[0] # longitudinal speed in vehicle coordinate
    x = original_states[1] # longitudinal position in road coordinate
    y = original_states[2] # lateral position in road coordinate
    phi = original_states[3] # vehicle heading in absolute coordinate
    phid = math.pi/2
    v = original_states[4] # lateral speed in vehicle coordinate
    r = original_states[5] # yaw rate in absolute coordinate
    whole_states = [original_states]
    Caf, Car = parameters["Caf"], parameters["Car"]
    a, b, m, Iz = parameters["a"], parameters["L"]-parameters["a"], parameters["m"], parameters["Iz"]
    # Use Runge–Kutta method
    k1 = helper_state_update(original_states, action, parameters)
    states_k2 = [original_states[i]+duration*k1[i]/2 for i in range(len(original_states))]
    k2 = helper_state_update(states_k2, action, parameters)
    states_k3 = [original_states[i]+duration*k2[i]/2 for i in range(len(original_states))]
    k3 = helper_state_update(states_k3, action, parameters)
    states_k4 = [original_states[i]+duration*k3[i] for i in range(len(original_states))]
    k4 = helper_state_update(states_k4, action, parameters)
    RK_states = [original_states[i]+duration*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6 for i in range(len(original_states))]
    # Euler method
    # dt = 0.0001
    # num_step = int(duration/dt)
    # remaining_time = duration - num_step*dt
    # for step in range(num_step):
    #     dudt = au
    #     dxdt = u*math.cos(phi-phid)-v*math.sin(phi-phid)
    #     dydt = v*math.cos(phi-phid)+u*math.sin(phi-phid)
    #     dphidt = r
    #     dvdt = -(Caf+Car)/(m*u)*v+((b*Car-a*Caf)/(m*u)-u)*r+(Caf/m)*deltaf
    #     drdt = (b*Car-a*Caf)/(Iz*u)*v-((a**2)*Caf+(b**2)*Car)/(Iz*u)*r+a*(Caf/Iz)*deltaf
    #     states = [u+dudt*dt, x+dxdt*dt, y-dydt*dt, phi+dphidt*dt, v+dvdt*dt, r+drdt*dt]
    #     whole_states.append(states)
    #     u,x,y,phi,v,r = states
    # if remaining_time > 0:
    #     dudt = au
    #     dxdt = u*math.cos(phi)-v*math.sin(phi)
    #     dydt = v*math.cos(phi)+u*math.sin(phi)
    #     dphidt = r
    #     dvdt = -(Caf+Car)/(m*u)*v+((b*Car-a*Caf)/(m*u)-u)*r+Caf/m*deltaf
    #     drdt = (b*Car-a*Caf)/(Iz*u)*v-(a^2*Caf+b^2*Car)/(Iz*u)*r+a*Caf/Iz*deltaf
    #     states = [u+dudt*remaining_time, x+dxdt*remaining_time, y-dydt*remaining_time, phi+dphidt*remaining_time, v+dvdt*remaining_time, r+drdt*remaining_time]
    #     whole_states.append(states)
    #     u,x,y,phi,v,r = states
    return RK_states
    

def helper_state_update(states, action, parameters):
    """Helper function to update vehicle states based on simple vehicle dynamic model (bicycle model) using Runge-Kutta method.

    Args:
        states (list): Vehicle states including longitudinal speed in vehicle coordinate, longitudinal position in road coordinate, lateral position in road coordinate, heading in absolute coordinate, lateral speed in vehicle coordinate, yaw rate in absolute coordinate.
        action (dict): Next action including longitudinal acceleration in vehicle coordinate and steering angle.
        parameters (dict): Vehicle dynamics parameters including a, L, m, Iz, Caf, Car.

    Returns:
        list: New vehicle states with the same format as the original states.
    """
    au = action["acceleration"] # longitudinal acceleration in vehicle coordinate 
    deltaf = action["steering_angle"]/180*math.pi # steering angle in vehicle coordinate
    u = states[0] # longitudinal speed in vehicle coordinate
    x = states[1] # longitudinal position in road coordinate
    y = states[2] # lateral position in road coordinate
    phi = states[3] # vehicle heading in absolute coordinate
    phid = math.pi/2
    v = states[4] # lateral speed in vehicle coordinate
    r = states[5] # yaw rate in absolute coordinate
    Caf, Car = parameters["Caf"], parameters["Car"]
    a, b, m, Iz = parameters["a"], parameters["L"]-parameters["a"], parameters["m"], parameters["Iz"]
    # Use Runge–Kutta method
    k = [
        au, 
        u*math.cos(phi-phid)-v*math.sin(phi-phid), 
        -(v*math.cos(phi-phid)+u*math.sin(phi-phid)),
        r, 
        -(Caf+Car)/(m*u)*v+((b*Car-a*Caf)/(m*u)-u)*r+(Caf/m)*deltaf, 
        (b*Car-a*Caf)/(Iz*u)*v-((a**2)*Caf+(b**2)*Car)/(Iz*u)*r+a*(Caf/Iz)*deltaf
    ]
    return k

def check_network_boundary(center_pos, restrictions, states, init_pos):
    """Check if the vehicle is within the network boundary.

    Args:
        center_pos (list): Center position of the vehicle.
        restrictions (list): Restrictions of the network boundary.
        states (list): Vehicle states.
        init_pos (list): Initial position of the vehicle.

    Returns:
        list: New center position of the vehicle.
        list: New vehicle states.
        bool: True if the vehicle is within the network boundary. Otherwise, False.
    """
    within_flag = False
    x, y = center_pos
    x_lim, y_lim = restrictions
    new_states = list(states)
    new_center_pos = list(center_pos)
    # now only consider straight road in x direction
    if y > y_lim[1]:
        # print("find")
        new_center_pos[1] = y_lim[1]
        new_states[2] = y_lim[1] - init_pos[1]
        new_states[3] = math.pi/2
        new_states[4] = 0.
        new_states[5] = 0.
    elif y < y_lim[0]:
        # print("find")
        new_center_pos[1] = y_lim[0]
        new_states[2] = y_lim[0] - init_pos[1]
        new_states[3] = math.pi/2
        new_states[4] = 0.
        new_states[5] = 0.
    else:
        within_flag = True
    return new_center_pos, new_states, within_flag

def check_vehicle_info(veh_id, subsciption, simulator):
    """Check the vehicle information.

    Args:
        veh_id (str): Vehicle ID.
        subsciption (dict): Vehicle subscription results.
        simulator (Simulator): Simulator object.

    Returns:
        dict: Updated vehicle subscription results.
    """
    # check lateral speed of BV in the beginning
    if veh_id != "CAV":
        heading = subsciption[67]
        v_lat = subsciption[50]
        if round(heading,2) != 90 and round(v_lat,2) == 0:
            # print(f"Zero lateral speed for heading!=90, so change lateral speed for {veh_id}!")
            subsciption[50] = (2*(heading > 90)-1)*(-4.)
        
    # check BV's angle especially during lane change for replay
    if veh_id != "CAV":
        heading = subsciption[67]
        if conf.train_mode == "offline":
            time_step = str(round(float(simulator.env.init_time_step)+simulator.sumo_time_stamp,1))
            if time_step in simulator.env.replay_trajs and veh_id in simulator.env.replay_trajs[time_step]:
                if heading != float(simulator.env.replay_trajs[time_step][veh_id]['angle']):
                    subsciption[67] = float(simulator.env.replay_trajs[time_step][veh_id]['angle'])
                    # print(f"Not the same heading for {veh_id} at {time_step}, change from {heading} to {subsciption[67]}!")
    
    if conf.train_mode == "offline" and simulator.sumo_time_stamp == 0 and veh_id in simulator.env.replay_trajs[simulator.env.init_time_step]:
        subsciption[114] = float(simulator.env.replay_trajs[simulator.env.init_time_step][veh_id]["acceleration"])
        
    return subsciption


if __name__=="__main__":
    new_state = update_vehicle_real_states(
        [
            40.0,
            249.1535907862714,
            -3.646629531188548,
            1.713139951764376,
            -1.3379077455130062,
            0.05819019272610548
        ],
        {"acceleration":0,"steering_angle":0.07598464465483168},
        {
            "L": 2.54, # wheel base (m)
            "a": 1.14, # distance c.g. to front axle (m)
            "m": 1500, # mass (kg)
            "Iz": 2420, # yaw moment of inertia (kg-m^2)
            "Caf": 44000*2, # cornering stiffness -- front axle (N/rad)
            "Car": 47000*2, # cornering stiffness -- rear axle (N/rad)
            "g": 9.81
        },
        0.1
    )
    print(new_state)
    # new_state = update_vehicle_real_states(
    #     new_state,
    #     {"acceleration":1.1,"steering_angle":0},
    #     {
    #         "L": 2.54, # wheel base (m)
    #         "a": 1.14, # distance c.g. to front axle (m)
    #         "m": 1500, # mass (kg)
    #         "Iz": 2420, # yaw moment of inertia (kg-m^2)
    #         "Caf": 44000*2, # cornering stiffness -- front axle (N/rad)
    #         "Car": 47000*2, # cornering stiffness -- rear axle (N/rad)
    #         "g": 9.81
    #     },
    #     0.1
    # )
    # print(new_state)