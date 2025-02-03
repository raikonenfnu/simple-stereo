import numpy as np
import torch
import matplotlib.pyplot as plt


############ Plotting tools ##############
def _epipoles(E):
    U, S, V = np.linalg.svd(E)
    e1 = V[-1, :]
    U, S, V = np.linalg.svd(E.T)
    e2 = V[-1, :]
    return e1, e2

def displayEpipolarF(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, timeout=3600, mouse_stop=2)[0]

        xc = x
        yc = y
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)
        plt.draw()

#########################

def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))
    return F


def eightpoint(pts1, pts2, transform1, transform2):
    """
    Formulate coplanarity constraint
    x'.T * F * x'' = 0

    Into system of linear equation
    Af = 0

    Where F = fundamental matrix
    [[F11, F12, F13],
     [F21, F22, F23],
     [F31, F32, F33]]
    x' = homogenous coord on camera1 [X', Y', 1]
    x'' = homogenous coord on camera2 [X'', Y'', 1]

    On a single point/correspondant, we will get:

    [X'*F11 + Y' * F21 + F31]
    [X'*F12 + Y' * F22 + F32] * [X'', Y'', 1] = 0
    [X'*F13 + Y' * F23 + F33]

    Simplifying to
    X'*F11 * X'' + Y' * F21 * X'' + F31  * X'' +
    X'*F12 * Y'' + Y' * F22 * Y'' + F32 * Y'' +
    X'*F13 + Y' * F23 + F33
    = 0 

    We can formulate this as system of linear equation:
    [X'*X'', Y' * X'', X'', X' * Y'', Y' * Y'', Y'', X', Y', 1] *
    [F11, F21, F31, F12, F22, F23, F13, F23, F33] = 0

    Hence staciking 8 of the lhs operands/points we can apply SVD
    and get the solution for F.
    """
    assert pts1.shape[-1] == 3, "Only handle homogenous 3D coord."
    assert pts1.shape == pts2.shape, "Expect correspondants to have same shape."

    # Construct system of linear equations.
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    A_linear_equation = torch.vstack((x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, torch.ones(x1.shape))).T

    # Compute SVD to solve linear equation
    svd_res = torch.linalg.svd(A_linear_equation)
    # Extract the rightmost singular vector (corresponding to the smallest singular value)
    # Since smallest singular value/vector correspond to being closest to null space.
    fundamental_vector = svd_res.Vh[-1, :]

    # Reshape and reorder to get fundamental matrix.
    fundamental_matrix = fundamental_vector.reshape(3,3).T

    # Singularize by making SVD value 0 for better numerical stability.
    fundamental_matrix = _singularize(fundamental_matrix)

    # Unscale fundamental matrix by transforms.
    # With scaling our coplanarity looks like
    # x_normalized'.T * F * x_normalized'' = 0
    # where x_normalized' = T' @ x', similar for x''.
    # =>  x'.T * T'.T * F * T'' * x''
    # => To scale it properly we need: F = T'.T * F * T''
    fundamental_matrix = transform1.T @  fundamental_matrix @ transform2
    return fundamental_matrix


def essentialMatrix(fundamental_matrix, camera_K1_intrinsic, camera_K2_intrinsic):
    """
    From coplanarity constraint:
    x_uncalibrated'.T * F * x_uncalibrated'' = 0
    where x_calibrated = K * x_uncalibrated
    =>  x_calibrated'.T * K1.T * F * K2 * x_calibrated'' = 0
    We also know x_calibrated'.T * E * x_calibrated'' = 0
    => E = K1.T * F * K2
    """
    return camera_K1_intrinsic.T  @ fundamental_matrix @ camera_K2_intrinsic

def preprocess_points(pts):
    """
    Normalizes point values to be between 0~1 by dividing
    by X-coordinate by max of X, and y-coord by max of Y.
    This function also turns points into homogenous points 
    [X,Y] -> [X,Y, 1].
    """
    assert pts.shape[-1] == 2, "Only handle non homogenous 2D coord."

    # Homogenize coordinates [X, Y] -> [X, Y, 1]
    ones = torch.ones(pts.shape[0], 1, dtype=pts.dtype)
    homogenous_pts = torch.hstack([pts, ones]).to(torch.float32)

    # Normalize coordinate values to be between 0 and 1.
    # by dividing x-coords by max(X) and y-coord by max(Y)
    x_max = torch.max(pts[:,0])
    y_max = torch.max(pts[:,1])

    # Represent simple pts[:,0] = pts[:,0] /x_max
    # and pts[:,1] = pts[:,1] /y_max as a transformation
    # matrix. This is because we'd need this in transformation
    # matrix form to un-normalize/rescale fundamental matrix.
    scale_x = 1.0 / x_max
    scale_y = 1.0 / y_max
    scaling_transform = torch.eye(3)
    scaling_transform[0,0] = scale_x
    scaling_transform[1,1] = scale_y
    normalized_pts = torch.einsum('ik,jk->ji', scaling_transform, homogenous_pts)

    return normalized_pts, scaling_transform


def main():
    data = np.load("data/temple/some_corresp.npz")
    camera1_pts = torch.from_numpy(data["pts1"]).to(torch.int32)
    camera2_pts = torch.from_numpy(data["pts2"]).to(torch.int32)
    camera1_pts, camera1_transform = preprocess_points(camera1_pts)
    camera2_pts, camera2_transform = preprocess_points(camera2_pts)
    fundamental_matrix = eightpoint(camera1_pts, camera2_pts, camera1_transform, camera2_transform)
    print("Fundamental matrix:", fundamental_matrix)
    # im1 = plt.imread('data/temple/im1.png')
    # im2 = plt.imread('data/temple/im2.png')
    # displayEpipolarF(im1, im2, fundamental_matrix.numpy())

    # Load camera intrinsic
    intrinsics = np.load('data/temple/intrinsics.npz')
    camera_K1_intrinsic = torch.from_numpy(intrinsics["K1"]).to(torch.float32)
    camera_K2_intrinsic = torch.from_numpy(intrinsics["K2"]).to(torch.float32)
    essential_matrix = essentialMatrix(fundamental_matrix, camera_K1_intrinsic, camera_K2_intrinsic)
    print("Essential matrix:", essential_matrix)

if __name__ == "__main__":
    main()