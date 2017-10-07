/* ray-tracer.cpp

Ray Tracing Image Renderer
by Dylan Quenneville

Developed with guidance from Christopher Andrews

Borrowed some code from Brandon Jones and Colin MacKenzie IV's gl-matrix.js

*/

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <list>
#include <vector>
#include <math.h>
#include "imageLib.h"

#define PI 3.14159265
#define SHADOW_T0 0.00001
#define REFLECT_T0 0.00001
#define MIN(a, b) ((a) <= (b) ? (a) : (b))
#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#define SMALLEST_POSITIVE(a, b) ((a) < 0 ? (b) : ((b) < 0 ? (a) : MIN(a, b)))

enum shape_type { plane, sphere, triangle, detailed_triangle};

class Vec3;
class Ray;
class Mat4;

class Mat4 {
	public:
	double data[16];
	Mat4() {
		for (int i = 0; i < 16; i++) {
			data[i] = 0.0;
		}
		data[0] = 1;
		data[5] = 1;
		data[10] = 1;
		data[15] = 1;
	}
	Mat4(double new_data[16]) {
		for (int i = 0; i < 16; i++) {
			data[i] = new_data[i];
		}
	};
	double operator[](int index) {
		return data[index];
	};
	void add(Mat4 matrix2) {
		for (int i = 0; i < 16; i++) {
			data[i] += matrix2[i];
		}
	};
	void subtract(Mat4 matrix2) {
		for (int i = 0; i < 16; i++) {
			data[i] -= matrix2[i];
		}
	};
	// code borrowed from Brandon Jones, Colin MacKenzie IV's
	// gl-matrix.js
	static void multiply(double out[16], double a[16], double b[16]) {
    double a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3],
        a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7],
        a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11],
        a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

    // Cache only the current line of the second matrix
    double b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    out[0] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[1] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[2] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[3] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

    b0 = b[4]; b1 = b[5]; b2 = b[6]; b3 = b[7];
    out[4] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[5] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[6] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[7] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

    b0 = b[8]; b1 = b[9]; b2 = b[10]; b3 = b[11];
    out[8] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[9] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[10] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[11] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

    b0 = b[12]; b1 = b[13]; b2 = b[14]; b3 = b[15];
    out[12] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[13] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[14] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[15] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
	};
	// void normalize() {
	// 	double a = x*x + y*y + z*z;
	// 	if (a > 0) {
	// 		a = 1 / sqrt(a);
	// 		scale(a);
	// 	}
	// };
};


Vec3 pointFromRay(Ray ray, double t);

Vec3 normalFromSphere(Vec3 center, Vec3 surface_point);

class Vec3 {
	public:
	double x, y, z;
	Vec3() {
		x = 0.0;
		y = 0.0;
		z = 0.0;
	}
	Vec3(double new_x, double new_y, double new_z) {
		x = new_x;
		y = new_y;
		z = new_z;
	};
	void add(Vec3 vector2) {
		x += vector2.x;
		y += vector2.y;
		z += vector2.z;
	};

	// subtract
	void subtract(Vec3 vector2) {
		x -= vector2.x;
		y -= vector2.y;
		z -= vector2.z;
	};
	static Vec3 subtract(Vec3 v1, Vec3 v2) {
		return Vec3(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);
	};
	static void subtract(Vec3 &output, Vec3 v1, Vec3 v2) {
		output = Vec3(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);
	};
	void scale(double s) {
		x *= s;
		y *= s;
		z *= s;
	};
	void multiply(Vec3 v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
	};
	static Vec3 multiply(Vec3 v1, Vec3 v2) {
		return Vec3(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);
	};
	static void multiply(Vec3 &output, Vec3 v1, Vec3 v2) {
		output = Vec3(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);
	};
	void multiply(Mat4 m) {
		x = m[0]*x + m[1]*y + m[2]*z + m[3];
		y = m[4]*x + m[5]*y + m[6]*z + m[7];
		z = m[8]*x + m[9]*y + m[10]*z + m[11];
	};
	double dot(Vec3 vector2) {
		return x*vector2.x + y*vector2.y + z*vector2.z;
	};
	void normalize() {
		double a = x*x + y*y + z*z;
		if (a > 0) {
			a = 1 / sqrt(a);
			scale(a);
		}
	};
};



void logVec3(Vec3 v) {
	// cout << "vec3: (" << v.x << ", " << v.y << ", " << v.z << ")\n";
}

class Ray {
	public:
	Vec3 p, v, pixel;
	Ray(Vec3 point, Vec3 vec) {
		p = point;
		v = vec;
	}
};

class Hit {
	public:
	int type, success;
	double t, reflectivity;
	Vec3 p, normal, color;
	Hit(){
		t = -1;
		success = 0;
		color = Vec3(0.5, 0.1, 0.8);
	};
	Hit(int a) { // really just used to make success 0
		t = a;
		success = a;
		color = Vec3(0.5, 0.1, 0.8);
	}
	Hit(double new_t, Vec3 new_p, Vec3 new_n, Vec3 new_c, double new_r) {
		t = new_t;
		p = new_p;
		normal = new_n;
		color = new_c;
		reflectivity = new_r;
		success = 1;
	}
	Hit(double new_t, Vec3 new_p, Vec3 new_n, Vec3 new_c, double new_r, int new_type) {
		t = new_t;
		p = new_p;
		normal = new_n;
		color = new_c;
		reflectivity = new_r;
		type = new_type;
		success = 1;
	}
	int valid() {
		return success > 0;
	}
};

float smallestPositive(float a, float b) {
  if (a < 0.0)
    return b;
  else if (b < 0.0)
    return a;
  else
    return min(a, b);
}

class Shape {
	public:
	int type;
	Vec3 p1, p2, p3;
	Vec3 n1, n2, n3;
	Vec3 c1, c2, c3;
	double radius, reflectivity;
	Vec3 point() {return p1;};
	Vec3 normal() {return n1;};
	Vec3 color() {return c1;};
	Hit hitPlane(Ray ray) {
		Vec3 result = p1;
		result.subtract(ray.p);
		if (n1.dot(ray.v) == 0.0) {
			return Hit(-1);
		}
		double t = n1.dot(result)/n1.dot(ray.v);
		Vec3 hit_point = pointFromRay(ray, t);
		return Hit(t, hit_point, n1, c1, reflectivity, 0);
	};
	Hit hitSphere(Ray ray) {
		double t, t1, t2, a, b, c;
		Vec3 eye_to_center = ray.p;
		eye_to_center.subtract(p1);
		a = ray.v.dot(ray.v);
		if (a == 0.0) {
			return Hit(-1);
		}
		b = ray.v.dot(eye_to_center);
		c = eye_to_center.dot(eye_to_center) - radius*radius;

		t1 = (-b + sqrt(b*b - a*c))/a;
		t2 = (-b - sqrt(b*b - a*c))/a;
		t = SMALLEST_POSITIVE(t1, t2);
		Vec3 hit_point = pointFromRay(ray, t);
		return Hit(t, hit_point, normalFromSphere(p1, hit_point), c1, reflectivity, 1);
	};
	Hit hitTriangle(Ray ray) {
		double a = p1.x-p2.x,
			b = p1.y - p2.y,
			c = p1.z - p2.z,
			d = p1.x - p3.x,
			e = p1.y - p3.y,
			f = p1.z - p3.z,
			g = ray.v.x,
			h = ray.v.y,
			i = ray.v.z,
			j = p1.x - ray.p.x,
			k = p1.y - ray.p.y,
			l = p1.z - ray.p.z;

		double detM = a * (e * i - h*f) + d * (h*c - b*i) + g*(b*f - e*c);

		double gamma = (a * (k*i - h*l) + j * (h*c - b*i) + g * (b*l-k*c))/detM;
    if (gamma < 0.0 || gamma > 1.0){
      return Hit(-1);
    }

    double beta = (j * (e*i - h*f) + d * (h*l - k*i) + g * (k*f-e*l))/detM;
    if (beta < 0.0 || beta > 1.0){
      return Hit(-1);
    }

    double alpha = 1 - beta - gamma;
    if (alpha < 0.0 || alpha > 1.0){
      return Hit(-1);
    }

		double t = (a * (e*l - k*f) + d*(k*c-b*l) + j*(b*f-e*c))/detM;

		if (t > 0.0){
      Vec3 point = ray.v;
			point.scale(t);
      point.add(ray.p);

			return Hit(t, point, n1, c1, reflectivity, 2);
      // return {t:t, normal:normal, p: point, color: color, reflectivity: reflectivity};
    }else{
      return Hit();
    }


	};
	Hit hit(Ray ray) {
		switch (type) {
			case 0: // plane
				return hitPlane(ray);
				break;
			case 1: // sphere
				return hitSphere(ray);
				break;
			case 2: // sphere
				return hitTriangle(ray);
				break;
		}
		return Hit(-1);
	};
	Shape() {};
	Shape(int t, Vec3 v1, Vec3 v2, Vec3 v3, double refl) {
		if (t == 0) {// plane
			type = t;
			p1 = v1;
			n1 = v2;
			c1 = v3;
			radius = 0;
		}
		reflectivity = refl;
	};
	Shape(int t, Vec3 v1, double r, Vec3 v2, double refl) {
		if (t == 1) { // sphere
			type = t;
			p1 = v1;
			radius = r;
			c1 = v2;
		}
		reflectivity = refl;
	}
	Shape(int t, Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4, double refl) {
		if (t == 2) { // triangle
			type = t;
			p1 = v1;
			p2 = v2;
			p3 = v3;
			c1 = v4;
			Vec3 p2p1 = Vec3::subtract(p2, p1);
			Vec3 p3p1 = Vec3::subtract(p3, p1);

			Vec3 normal(p2p1.y*p3p1.z - p2p1.z*p3p1.y,
									p2p1.z*p3p1.x - p2p1.x*p3p1.z,
									p2p1.x*p3p1.y - p2p1.y*p3p1.x);
			normal.normalize();
			n1 = normal;
			n2 = normal;
			n3 = normal;
			logVec3(n1);
		}
		reflectivity = refl;
	}
};

class Light {
	public:
	Vec3 ambient, diffuse, specular, position;
	double shininess;
	Vec3 getAmbient(Vec3 color) {
		return Vec3::multiply(color, ambient);
	};
	Vec3 getPhong(Vec3 p, Vec3 normal, Vec3 camera, Vec3 color, double reflectivity) {
		Vec3 v = camera;
		v.subtract(p);
		v.normalize();
		Vec3 l = position;
		l.subtract(p);
		l.normalize();
		Vec3 h = l;
		h.add(v);
		h.normalize();

		Vec3 amb = color;
		amb.multiply(ambient);

		Vec3 diff = diffuse;
		diff.scale(MAX(l.dot(normal), 0.0));
		diff.multiply(color);

		Vec3 spec = specular;
		spec.scale(reflectivity*MAX(pow(MAX(normal.dot(h), 0.0), shininess), 0.0));

		Vec3 output(0.0, 0.0, 0.0);
		output.add(amb);
		output.add(diff);
		output.add(spec);
		return output;
	};
	Light() {};
	Light(Vec3 pos) {
		position = pos;
		ambient = Vec3(0.2, 0.2, 0.2);
		diffuse = Vec3(0.7, 0.7, 0.7);
		specular = Vec3(0.7, 0.7, 0.7);
		shininess = 300.0;
		cout << "ambient (" << ambient.x << ", " << ambient.y << ", " << ambient.z << ")\n";
		cout << "position (" << position.x << ", " << position.y << ", " << position.z << ")\n";
	};
	// Light(Vec3 pos, Vec3 amb, Vec3 diff, Vec3 spec, double shiny) {
	// 	position = pos;
	// 	ambient = amb;
	// 	diffuse = diff;
	// 	specular = spec;
	// 	shininess = shiny;
	// }
};


class View {
	public:
	Vec3 camera;
	double n, l, r, t, b;
	View() {}
	View(Vec3 position, double aspect, double angle) {
		camera = position;
		n = -1.0;
		l = -aspect*tan(angle/2);
		r = aspect*tan(angle/2);
		t = tan(angle/2);
		b = -tan(angle/2);
	}
};

class Scene {
	public:
	Light lighting;
	View view;
	list<Shape> shapes;
	Hit findHit(Ray ray, double t0, double t1) {
		double best_t = INFINITY;
		Hit best_hit, next_hit;
		for (list<Shape>::iterator shape_it = shapes.begin(); shape_it != shapes.end(); ++shape_it) {
			next_hit = shape_it->hit(ray);
			if (next_hit.valid() && next_hit.t >= t0 && next_hit.t <= t1 && next_hit.t < best_t) {
				best_hit = next_hit;
				best_t = next_hit.t;
			}
		}
		return best_hit;
	};
	Vec3 findColor(Ray ray, double t0, double t1, double depth) {
    if (depth <= 0) {
      return Vec3(0.0, 0.0, 0.0);
    }
		Hit best_hit = findHit(ray, t0, t1);
		if (best_hit.valid()) {
			double reflectivity = best_hit.reflectivity;
			Vec3 shadow_vector;
			shadow_vector = lighting.position;
			shadow_vector.subtract(best_hit.p);
			Ray shadow(best_hit.p, shadow_vector);
			Hit block_hit = findHit(shadow, SHADOW_T0, 1.0);
			Vec3 color;
			if (block_hit.valid()) { // there is an obstruction for lighting
				color = lighting.getAmbient(best_hit.color);
			} else {
				color = lighting.getPhong(best_hit.p, best_hit.normal,
						view.camera, best_hit.color, best_hit.reflectivity);
			}

			if (reflectivity > 0.0) {
				Vec3 reflection_vector = best_hit.normal;
				reflection_vector.scale(2*best_hit.normal.dot(ray.v));
				reflection_vector.subtract(ray.v);
				reflection_vector.scale(-1.0);
				reflection_vector.normalize();
				Ray reflection_ray(best_hit.p, reflection_vector);
				Vec3 reflection_color = findColor(reflection_ray, REFLECT_T0, INFINITY, depth-1);
				reflection_color.scale(0.2*reflectivity);
				color.scale(0.9);
				color.add(reflection_color);
			}


			return color;
		} else {
			return Vec3(0.0, 0.0, 0.0);
		}
	};
	Scene(View new_view, Light new_lighting){
		lighting = new_lighting;
		view = new_view;
	};
};


Ray rayFromPixel(double pixel_x, double pixel_y, int w, int h, View v) {
	double real_x = (pixel_x) / double(w) * (v.r-v.l) + v.l;
	double real_y = v.t - (pixel_y) / double(h) * (v.t-v.b);
	// if (pixel_y == 250) {
	// 	cout << real_y << '\n';
	// }
	Ray output(v.camera, Vec3(real_x, real_y, v.n));
	output.pixel = Vec3(pixel_x, pixel_y, 0);
	return output;
	return Ray(v.camera, Vec3(real_x, real_y, v.n));
}


Vec3 pointFromRay(Ray ray, double t) {
	Vec3 point = ray.v;
	point.scale(t);
	point.add(ray.p);
	return point;
}

Vec3 normalFromSphere(Vec3 center, Vec3 surface_point) {
	surface_point.subtract(center);
	surface_point.normalize();
	return surface_point;
}




// Clears image and z-buffer.
void initImage(CByteImage &canvas, int w, int h) {

	for (int x = 0; x < w; ++x) {
		for (int y = 0; y < h; ++y) {
			canvas.Pixel(x,y,0) = 255;
			canvas.Pixel(x,y,1) = 255;
			canvas.Pixel(x,y,2) = 255;
			canvas.Pixel(x,y,3) = 255;
		}
	}
}


void render(CByteImage &canvas, Scene &scene, int w, int h, double subpixel, int depth) {
	Vec3 color(0.0, 0.0, 0.0);
	for (double y = 0.0; y < h; y++) {
		for (double x = 0.0; x < w; x++) {
			if (subpixel > 1.0) {
				color = Vec3(0.0, 0.0, 0.0);
				for (double xsub = 0.0; xsub < subpixel; xsub += 1.0) {
					for (double ysub = 0.0; ysub < subpixel; ysub += 1.0) {
						color.add(scene.findColor(rayFromPixel(double(x)+xsub/subpixel, double(y)+ysub/subpixel, w, h, scene.view), 0, INFINITY, depth));
					}
				}
				color.scale(1.0/(subpixel*subpixel));
			} else {
				color = scene.findColor(rayFromPixel(x+0.5, y+0.5, w, h, scene.view), 0, INFINITY, depth);
			}
			canvas.Pixel(x,y,0) = MIN(int(color.z*255.0), 255); // B
			canvas.Pixel(x,y,1) = MIN(int(color.y*255.0), 255); // G
			canvas.Pixel(x,y,2) = MIN(int(color.x*255.0), 255); // R
			canvas.Pixel(x,y,3) = 255;
		}
		cout << "row " << y << " of " << h << " completed\n";
	}
}

void eraseComments(ifstream &input) {
	while (input.peek() == ' ' || input.peek() == '\n' || input.peek() == '#')
		if (input.peek() == ' ' || input.peek() == '\n')
			input.get();
		else if (input.peek() == '#')	// if there is a comment ignore rest of the line
			input.ignore(800, '\n');
}

// scan .obj file and add faces accordingly
void loadOBJ(string filename, list<Shape> &shapes, Mat4 transformation,
		Vec3 color, double reflectivity) {
	cout << "loading " << filename << '\n';
	ifstream input (filename.c_str());
	char info_type;
	double x, y, z;
	int a, b, c;
	vector<Vec3> vertices;
	list<int> indices;
	Vec3 new_vertex;

	vertices.push_back(Vec3(0, 0, 0));
	while (!input.eof()) {
		if (input.peek() == '\n') {
			input.get();
			// count++;
			continue;
		} else if (input.peek() == EOF) {
			break;
		}
		input >> info_type;
		if (info_type == 'v') {
			input >> x >> y >> z;
			new_vertex = Vec3(x, z, -y);
			// logVec3(new_vertex);
			new_vertex.multiply(transformation);
			// cout << " -> becomes ";
			// logVec3(new_vertex);
			vertices.push_back(new_vertex);
		} else if (info_type == 'f') {
			input >> a >> b >> c;
			indices.push_back(a);
			indices.push_back(b);
			indices.push_back(c);
			// cout << "index: ";
			// logVec3(Vec3(a, b, c));
		}
		// cout << "info type" << info_type << '\n';

	}
	Vec3 p1, p2, p3;
	// cout << "loaded " << indices.size() << " indices\n";
	// cout << "loaded " << vertices.size() << " vertices\n";
	int n_indices = (int)indices.size() / 3;
	for (int i = 0; i < n_indices; i++) {
		p1 = vertices[indices.front()];
		indices.pop_front();
		p2 = vertices[indices.front()];
		indices.pop_front();
		p3 = vertices[indices.front()];
		indices.pop_front();
		// cout << "adding triangle with coordinates:\n";
		// logVec3(p1);
		// logVec3(p2);
		// logVec3(p3);
		shapes.push_back(Shape(2, p1, p2, p3, color, reflectivity));
	}
}

void loadShapes(ifstream &input, list<Shape> &shapes) {
	int count = 0;

	// data loading variables
	double x, y, z, nx, ny, nz, r, g, b, radius, reflectivity;
	double x1, x2, x3, y1, y2, y3, z1, z2, z3;
	Vec3 p1, p2, p3;
	Vec3 n1, n2, n3;
	Vec3 c1, c2, c3;
	double transform[16] = {1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, -2.0, 0, 0, 0, 1};;
	string obj_filename;

	// load shapes
	while (!input.eof()) {
		if (input.peek() == '\n') {
			input.get();
			// count++;
			continue;
		} else if (input.peek() == EOF) {
			break;
		} else if (count > 1000) {
			throw CError("\n  usage: too many shapes loaded or incorrect formatting\n");
		}
		eraseComments(input);
		int shape_type;
		input >> shape_type;

		Shape new_shape;

		if (shape_type == 0) { // plane
			input >> x >> y >> z >> nx >> ny >> nz >> r >> g >> b >> reflectivity;
			p1 = Vec3(x, y, z);
			n1 = Vec3(nx, ny, nz);
			c1 = Vec3(r, g, b);
			new_shape = Shape(0, p1, n1, c1, reflectivity);
		} else if (shape_type == 1) { // sphere
			input >> x >> y >> z >> radius >> r >> g >> b >> reflectivity;
			p1 = Vec3(x, y, z);
			c1 = Vec3(r, g, b);
			new_shape = Shape(1, p1, radius, c1, reflectivity);
		} else if (shape_type == 2) { // triangle
			input >> x1 >> y1 >> z1 >> x2 >> y2 >> z2 >> x3 >> y3 >> z3 >> r >> g >> b >> reflectivity;
			p1 = Vec3(x1, y1, z1);
			p2 = Vec3(x2, y2, z2);
			p3 = Vec3(x3, y3, z3);
			c1 = Vec3(r, g, b);
			new_shape = Shape(2, p1, p2, p3, c1, reflectivity);
		} else if (shape_type == 3) { // .obj file
			input >> obj_filename >> r >> g >> b >> reflectivity;
			loadOBJ(obj_filename, shapes, Mat4(transform), Vec3(r, g, b), reflectivity);
			new_shape.type = 3;
		}

		printf("added shape of type %d\n", new_shape.type);
		shapes.push_back(new_shape);
		count++;
	}
}


int main(int argc, char *argv[]) {

	try {
		if (argc < 3 || argc > 5) {
			throw CError("\n  usage: %s scene_info.txt output.png [subpixel-accuracy reflection-depth]\n", argv[0]);
		}

	ifstream input (argv[1]);
	char* outname = argv[2];

	double subpixel;
	if (argc > 3) {
		subpixel = atof(argv[3]);
	} else {
		subpixel = 1.0;
	}
	double depth;
	if (argc > 4) {
		depth = atof(argv[4]);
	} else {
		depth = 2;
	}

	eraseComments(input);

	int width, height;
	double cam_x, cam_y, cam_z, cam_angle, aspect;//, near;
	double light_x, light_y, light_z;
	string name;
	// Read in dimensions and # shapes/views
	input >> width >> height;

	eraseComments(input);
	input >> cam_x >> cam_y >> cam_z >> cam_angle;
	eraseComments(input);
	input >> name;

	eraseComments(input);
	input >> light_x >> light_y >> light_z;

	View temp_view(Vec3(cam_x, cam_y, cam_z), double(width)/double(height), PI/3);
	// Light ;
	Scene scene(temp_view, Light(Vec3(light_x, light_y, light_z)));

	cout << "ambient" << scene.lighting.ambient.x << scene.lighting.ambient.y << scene.lighting.ambient.z << '\n';

	loadShapes(input, scene.shapes);


	CByteImage canvas;

	CShape canvas_size(width, height, 4);
	canvas.ReAllocate(canvas_size);
	initImage(canvas, width, height);

	render(canvas, scene, width, height, subpixel, depth);
	/*
	mkdir(OUTPUT_FOLDER, 0700);
	mkdir((OUTPUT_FOLDER + subfolder_str + "/").c_str(), 0700);
	mkdir((OUTPUT_FOLDER + folder_name).c_str(), 0700);
	*/

	WriteImageVerb(canvas, outname, 1);
	input.close();
	}
	catch (CError &err) {
	fprintf(stderr, err.message);
	fprintf(stderr, "\n");
	return -1;
    }

    return 0;
}
