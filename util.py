import numpy as np
import os
import pickle
import pandas as pd

def read_skeleton_file(filename):
    '''
    translated form the matlab script
    '''
    with open(filename, 'r') as file:
        framecount = int(file.readline().split()[0])
        # for each frame, store skeleton info as an element in bodyinfo 
        bodyinfo=[]
        for f in range(framecount):
            bodycount = int(file.readline().split()[0])
            bodies = [] # this list contains all skeletons in a frame
            for b in range(bodycount):
                # create a dictionary to store all the relevent info of a skeleton in `body`
                body = {}
                body_raw = file.readline().split()
                body['bodyID'] = int(body_raw[0])
                body['clipedEdges'] = int(body_raw[1])
                body['handLeftConfidence'] = int(body_raw[2])
                body['handLeftState'] = int(body_raw[3])
                body['handRightConfidence'] = int(body_raw[4])
                body['handRightState'] = int(body_raw[5])
                body['isResticted'] = int(body_raw[6])
                body['leanX'] = float(body_raw[7])
                body['leanY'] = float(body_raw[8])
                body['trackingState'] = int(body_raw[9])
                
                body['jointCount'] = int(file.readline().split()[0])
                joints=[]
                for j in range(body['jointCount']):
                    jointinfo = [float(x) for x in file.readline().split()]
                    joint = {}

                    # 3D location of the joint j
                    joint['x'] = jointinfo[0]
                    joint['y'] = jointinfo[1]
                    joint['z'] = jointinfo[2]

                    # 2D location of the joint j in corresponding depth/IR frame
                    joint['depthX'] = jointinfo[3]
                    joint['depthY'] = jointinfo[4]

                    # 2D location of the joint j in corresponding RGB frame
                    joint['colorX'] = jointinfo[5]
                    joint['colorY'] = jointinfo[6]

                    # The orientation of the joint j
                    joint['orientationW'] = jointinfo[7]
                    joint['orientationX'] = jointinfo[8]
                    joint['orientationY'] = jointinfo[9]
                    joint['orientationZ'] = jointinfo[10]

                    # The tracking state of the joint j
                    joint['trackingState'] = int(jointinfo[11])

                    joints.append(joint)
                body['joints'] = joints
                bodies.append(body)
            bodyinfo.append(bodies)
        return bodyinfo

def normalize_joints(joints):
    """
    takes the ['joints'] field of a body dictionary, ie bodyinfo[3][0]['joints']
    output the list of 25 joints that are normalized, with xyz collected into nunmpy
    arrays.
    """
    neck = np.array([joints[2]['x'], joints[2]['y'], joints[2]['z']]).T
    left_hip = np.array([joints[12]['x'], joints[12]['y'], joints[12]['z']]).T
    right_hip = np.array([joints[16]['x'], joints[16]['y'], joints[16]['z']]).T
    
    u1 = right_hip - neck
    s = np.linalg.norm(u1, ord=2)
    u1 /= s
    
    u2_temp = left_hip - neck
    u3 = np.cross(u1, u2_temp)
    u3 /= np.linalg.norm(u3, ord=2)
    
    u2 = np.cross(u1, u3)
    
    basis = np.array([u1, u2, u3])
    
    scaled_joints = []
    for j in joints:
        coordinate = np.array([j['x'], j['y'], j['z']]).T
        relative_coordinate = (coordinate-neck)
        normalized_coordinate = np.matmul(basis, relative_coordinate) / s
        scaled_joints.append(normalized_coordinate)
    
    return scaled_joints

def process_skeleton(sfile, skeletons_dir):
    '''
    process a .skeleton file into a .pickle file. Saves to the same directory
    
    Pickle files format:

    the processed_bodyinfo is a numpy array that gets dumped into a pickle is in the following format:
    
    bodyinfo[f][b][j][u]
    Where: 
        f is the frame number             (depends on length of the video)
        b is the body number              (there might not always be just 1 body/skeleton in a frame)
        j is the joint number for a body  (25)
        u is the basis number for a joint (3)
    '''
    bodyinfo = read_skeleton_file(sfile)
    basename = os.path.basename(sfile).split('.')[0]
    save_path = os.path.join(skeletons_dir, basename+'.pickle')
    processed_bodyinfo = []
    for frame in bodyinfo:
        processed_frame = []
        for body in frame:
            scaled_joints = normalize_joints(body['joints'])
            processed_frame.append(scaled_joints)
        processed_bodyinfo.append(processed_frame)
    with open(save_path, 'wb') as pfile:
        pickle.dump(np.asarray(processed_bodyinfo), pfile)

def pickle_to_action_array(pickle_file, keep_idx):
    '''
    Param: a converted pickle file of processed skeleton
    #Return: a n*25*3 array of a_t*sign(d_t). n is number of total actions, ie (frames-2)*bodys,
    #25 is for each joint, 3 is for xzy coordinate.
    Return: a [ (n+2)*13*3, (n+1)*13*3, n*13*3] * number_body list
    '''
    out = []
    with open(pickle_file, 'rb') as fp:
        bodyinfo = pickle.load(fp)
    if len(bodyinfo.shape) != 4:
        print(pickle_file)
    else:
        for body in range(bodyinfo.shape[1]):
            single_bodyinfo = bodyinfo[:, body, :, :]
            single_bodyinfo = single_bodyinfo[:, keep_idx, :]
            d = np.diff(single_bodyinfo, axis=0)
            a = np.diff(d, axis=0)
            out_body = [ single_bodyinfo, d, a, pickle_file]
            out.append(out_body)
    return out

def pickle_files_to_action_series(pickle_files, keep_idx):
    h = []
    for pf in pickle_files:
        h_single = pickle_to_action_array(pf, keep_idx)
        h.extend(h_single)
    return h