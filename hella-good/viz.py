from matplotlib import cm
from matplotlib import pyplot as plt
from mayavi import mlab
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def showHist(img):
	plt.hist(img.flatten(), bins=80, color='c')
	plt.xlabel("Hounsfield Units (HU)")
	plt.ylabel("Frequency")
	plt.show()


def mayaPlot(s, val):
	src = mlab.pipeline.scalar_field(s)
	mlab.pipeline.iso_surface(src, contours=[val, ], opacity=0.3)
	#mlab.pipeline.iso_surface(src, contours=[s.max()-0.1*s.ptp(), ],)
	mlab.show()


def plot_3d(image, threshold=-300):

	# Position the scan upright,
	# so the head of the patient would be at the top facing the camera
	p = image.transpose(2,1,0)

	verts, faces = measure.marching_cubes(p, threshold)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')

	# Fancy indexing: `verts[faces]` to generate a collection of triangles
	mesh = Poly3DCollection(verts[faces], alpha=0.1)
	face_color = [0.5, 0.5, 1]
	mesh.set_facecolor(face_color)
	ax.add_collection3d(mesh)

	ax.set_xlim(0, p.shape[0])
	ax.set_ylim(0, p.shape[1])
	ax.set_zlim(0, p.shape[2])

	plt.show()


def grey2color(img):
	c = cm.jet(img)
	return c



def plotWithColorbar(img, filename):
    fig, ax = plt.subplots()
    i = ax.imshow(img, interpolation='nearest')
    fig.colorbar(i)
    plt.savefig(filename)
    fig.clf()
    plt.close()


def plot_nodule(nodule_crop):
	# Learned from ArnavJain
	# https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
	f, plots = plt.subplots(int(nodule_crop.shape[0] / 4) + 1, 4, figsize=(10, 10))

	for z_ in range(nodule_crop.shape[0]):
		plots[int(z_ / 4), z_ % 4].imshow(nodule_crop[z_, :, :])

	# The last subplot has no image because there are only 19 images.
	plt.show()