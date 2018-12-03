import numpy as np

def read_skeleton_file(filename):
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
    u3 = np.cross(u1, u2_temp) / np.linalg.norm(u2_temp, ord=2)
    
    u2 = np.cross(u1, u3)
    
    basis = np.array([u1, u2, u3])
    
    scaled_joints = []
    for j in joints:
        coordinate = np.array([j['x'], j['y'], j['z']]).T
        relative_coordinate = (coordinate-neck)
        normalized_coordinate = np.matmul(basis, relative_coordinate) / s
        scaled_joints.append(normalized_coordinate)
    
    return scaled_joints