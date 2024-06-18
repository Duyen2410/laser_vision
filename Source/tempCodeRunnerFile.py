def extract_laser_point(thinned, fx, fy, cx, cy, rotation_matrix, laser_undis, tvec):
    pointinlaserplane = []
    rows, cols = laser_undis.shape
    line = LaserCenter(thinned)     
    inv = np.linalg.inv(rotation_matrix)
    for i in range(400,rows-200,1):
        for j in range(650,cols-650,1):
            if line[i][j] == 255:
                cv.circle(laser_undis, (j,i), 5, [0,255,0], 2)
                Zc = (tvec[0][0] * inv[2][0] +  tvec[1][0] * inv[2][1] + tvec[2][0] * inv[2][2])/(inv[2][0]/fx*(j-cx) + inv[2][1]/fy*(i-cy) + inv[2][2])
                C = np.array([Zc/fx*(j-cx), Zc/fy*(i-cy), Zc]).T.reshape(3,1)
                pointinlaserplane.append(C)  
    return pointinlaserplane