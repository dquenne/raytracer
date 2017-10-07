// ray tracer with triangles (for final exam)
// Dylan Quenneville for CS 461 Computer Graphics


shadow_t0 = 0.000001; // precision of how close a shadow obstacle can be

// create scene object to store all aspects of the scene
var createScene = function(view) {
  let lighting = createLight(vec4.fromValues(1.0, 2.0, 1.0, 1.0));
  console.log("lighting:", lighting);

  let background = [0.0, 0.0, 0.0];

  let objects = [];

  let findHit = function(ray, t0, t1) {
    let best_t = Infinity;
    let best_hit;
    let best_o = -1;
    for (let o = 0; o < objects.length; o++) {
      let hit = objects[o].hit(ray);
      if (hit.t >= t0 && hit.t <= t1 && hit.t < best_t) {
        best_o = o;
        best_hit = hit;
        best_t = hit.t;
      }
    }
    return best_hit;
  };

  return {
    addObject: (obj) => {
      objects.push(obj);
    },
    findColor: function(ray, t0, t1, depth) {
      if (depth == 0) {
        return background;
      }

      let best_hit = findHit(ray, t0, t1);

      if (best_hit) {

        // return best_hit.color;
        let reflectivity = best_hit.reflectivity;
        shadow_vector = vec4.create();
        vec4.subtract(shadow_vector, lighting.position, best_hit.p);
        shadow = createRay(best_hit.p, shadow_vector);
        block = findHit(shadow, shadow_t0, 1.0);
        let color;
        if (!block)
          color = lighting.getPhong(best_hit.p, best_hit.normal, view.camera, best_hit.color, best_hit.reflectivity);
        else {
          color = lighting.getAmbient(best_hit.color);
        }

        if (reflectivity > 0) { // if reflectivity is 0, don't calculate reflections
          let reflection_vector = vec4.create();
          vec4.scale(reflection_vector, best_hit.normal, 2*vec4.dot(best_hit.normal, ray.v));
          vec4.subtract(reflection_vector, ray.v, reflection_vector);
          vec4.normalize(reflection_vector, reflection_vector);
          let reflection_ray = createRay(best_hit.p, reflection_vector);
          let reflection_color = this.findColor(reflection_ray, 0.000001, Infinity, depth-1);

          vec4.scale(reflection_color, reflection_color, (0.2));
          vec4.scale(reflection_color, reflection_color, reflectivity);
          vec4.scale(color, color, 0.9)
          vec4.add(color, color, reflection_color);
        }

        return color;
      } else {
        return background;
      }
    }
  }
}

// create object to store view information
var createView = function(angle, aspect, near) {
  return {
    camera: vec4.fromValues(0.0, 0.0, 0.0, 1.0),
    n: -near,
    l: -aspect*near*Math.tan(angle/2),
    r: aspect*near*Math.tan(angle/2),
    t: near*Math.tan(angle/2),
    b: -near*Math.tan(angle/2)
  }
}

// create object to store lighting information
var createLight = function(position, ambient, diffuse, specular, shininess, reflectivity) {
  if (!ambient)
    ambient = vec3.fromValues(0.2, 0.2, 0.2);
  if (!diffuse)
    diffuse = vec3.fromValues(0.7, 0.7, 0.7);
  if (!specular)
    specular = vec3.fromValues(0.7, 0.7, 0.7);
  if (!shininess)
    shininess = 300.0;
  return {
    position: position,
    ambient: ambient,
    diffuse: diffuse,
    specular: specular,
    shininess: shininess,
    getAmbient: (color) => {
      let amb = vec4.create();
    	vec4.multiply(amb, color, ambient);
      return amb;
    },
    getPhong: (p, n, e, color) => {
      let V = vec4.create();
      vec4.subtract(V, e, p); // get V (view) vector
      vec4.normalize(V, V);
      let L = vec4.create();
      vec4.subtract(L, position, p); // get L (light) vector
      vec4.normalize(L, L);
      let H = vec4.create();
      vec4.add(H, L, V); // get H (light + view) vector
      vec4.normalize(H, H);

      let amb = vec4.create();
    	vec4.multiply(amb, color, ambient);

    	let diff = vec4.create();
      vec4.scale(diff, diffuse, Math.max(vec4.dot(L, n), 0.0));
      vec4.multiply(diff, color, diff);

    	let spec = vec4.create();
      vec4.scale(spec, specular, Math.max(Math.pow(Math.max(vec4.dot(n, H), 0.0), shininess), 0.0));
      // vec4.multiply(spec, color, spec);

      let output = vec4.create();
      vec4.add(output, amb, diff);
      vec4.add(output, output, spec);
      return output;
    }
  }
}


// create ray object
var createRay = function(point, vector) {
  return {
    p: point,
    v: vector
  }
}

// create 'hit' object
var createHit = function(t, point, normal, color, reflectivity) {
  return {
    t: t,
    p: point,
    normal: normal,
    color: color,
    reflectivity: reflectivity
  }
}

// create ray from camera (origin) to pixel (on canvas)
var rayFromPixel = function(pixel_x, pixel_y, w, h, view) {
  real_x = (pixel_x + 0.5) / w * (view.r-view.l) + view.l;
  real_y = view.t - (pixel_y + 0.5) / h * (view.t-view.b);
  return createRay(view.camera, vec4.fromValues(real_x, real_y, view.n, 0.0));
}

// get point in 3D space (x, y, z, 1) based on a ray and its t value
var pointFromRay = function(ray, t) {
  point = vec4.create();
  vec4.scale(point, ray.v, t);
  vec4.add(point, point, ray.p);
  // console.log(point);
  return point;
}


// create ray from camera (origin) to pixel (on canvas)
var createPlane = function(point, normal, color, reflectivity, color2) {
  vec4.normalize(normal, normal);
  let gridSize = 10;
  if (!color2)
    color2 = vec4.fromValues(0,0,0,1);
  let checkerColor = function(x, y) {
    if (Math.abs((Math.floor(x/2.0*gridSize) +
        Math.floor(y/2.0*gridSize)) % 2.0) == 1.0) {
      return color;
    } else {
      return color2;
    }
  }

  return {
    hit: (ray) => {
      result = vec4.create();
      vec4.subtract(result, point, ray.p);
      if (vec4.dot(normal, ray.v) == 0) {
        return false;
      }
      let t = (vec4.dot(normal, result)/vec4.dot(normal, ray.v));
      let hit_point = pointFromRay(ray, t)
      return createHit(t,
                       hit_point,
                       normal,
                       checkerColor(hit_point[0], hit_point[2]),
                       reflectivity);
    },
    color: color
  }
}

var smallestPositive = function(a, b) {
  if (a < 0)
    return b;
  else if (b < 0)
    return a;
  else
    return Math.min(a, b);
}

var normalFromSphere = function(center, surface_point) {
  let vector = vec4.create();
  vec4.subtract(vector, surface_point, center);
  vec4.normalize(vector, vector);
  return vector;
}

// create ray from camera (origin) to pixel (on canvas)
var createSphere = function(point, radius, color, reflectivity) {
  if (!reflectivity)
    reflectivity = 0.0;
  return {
    hit: (ray) => {
      let t1, t2;
      let e_minus_c = vec4.create();
      vec4.subtract(e_minus_c, ray.p, point);
      let a = vec4.dot(ray.v, ray.v);
      if (a == 0) {
        return false;
      }
      b = vec4.dot(ray.v, e_minus_c); // this is actually b/2, but the 2s cancel
      c = vec4.dot(e_minus_c, e_minus_c) - radius*radius;

      t1 = (-b + Math.sqrt(b*b - a*c))/a;
      t2 = (-b - Math.sqrt(b*b - a*c))/a;
      let t = smallestPositive(t1, t2);
      let hit_point = pointFromRay(ray, t);
      return createHit(t,
                       hit_point,
                       normalFromSphere(point, hit_point),
                       color,
                       reflectivity);
    },
    color: color
  }
}

// create a triangle for the ray tracer.
var createTriangle = function(p1, p2, p3, color, reflectivity = 0.0){

  let p2p1 = vec4.create();
  p2p1 = vec4.sub(p2p1, p2, p1);
  let p3p1 = vec4.create();
  p3p1 = vec4.sub(p3p1, p3, p1);

  var normal = vec4.fromValues(p2p1[1]*p3p1[2] - p2p1[2]*p3p1[1],
                              p2p1[2]*p3p1[0] - p2p1[0]*p3p1[2],
                              p2p1[0]*p3p1[1] - p2p1[1]*p3p1[0],
                              0);
  normal = vec4.normalize(normal, normal);

  let hit = (ray) =>{
    let a = p1[0] - p2[0],
        b = p1[1] - p2[1],
        c = p1[2] - p2[2],
        d = p1[0] - p3[0],
        e = p1[1] - p3[1],
        f = p1[2] - p3[2],
        g = ray.v[0],
        h = ray.v[1],
        i = ray.v[2],
        j = p1[0] - ray.p[0],
        k = p1[1] - ray.p[1],
        l = p1[2] - ray.p[2];


    let detM = a * (e * i - h*f) + d * (h*c - b*i) + g*(b*f - e*c);


    let gamma = (a * (k*i - h*l) + j * (h*c - b*i) + g * (b*l-k*c))/detM;
    if (gamma < 0 || gamma >1){
      return {t:-1};
    }


    let beta = (j * (e*i - h*f) + d * (h*l - k*i) + g * (k*f-e*l))/detM;
    if (beta < 0 || beta >1){
      return {t:-1};
    }


    let alpha = 1 - beta - gamma;
    if (alpha < 0 || alpha >1){
      return {t:-1};
    }


    let t = (a * (e*l - k*f) + d*(k*c-b*l) + j*(b*f-e*c))/detM;

    if (t > 0){
      let point = vec4.create();
      vec4.scale(point, ray.v, t);
      vec4.add(point, point, ray.p);

      return {t:t, normal:normal, p: point, color: color,
        reflectivity: reflectivity};
    }else{
      return {t:-1};
    }
  }

  return {
    hit:hit
  };
}


let setPixel = function(image, x,y,r,g,b){
  let offset = (image.width*y + x) * 4;
  image.data[offset] = r;
  image.data[offset+1] = g;
  image.data[offset+2] = b;
  image.data[offset+3] = 255;
};


// add spheres for a caterpillar
let addCaterpillar = function(scene) {
  scene.addObject(createSphere(vec4.fromValues(-0.225, -0.07, -0.78, 1.0), 0.12, vec4.fromValues(0.2, 0.9, 0.5, 1.0), 0.0));
  scene.addObject(createSphere(vec4.fromValues(-0.075, -0.1, -0.84, 1.0), 0.1, vec4.fromValues(0.2, 0.9, 0.5, 1.0), 0.0));
  scene.addObject(createSphere(vec4.fromValues(0.05, -0.1, -0.9, 1.0), 0.1, vec4.fromValues(0.2, 0.9, 0.5, 1.0), 0.0));
  scene.addObject(createSphere(vec4.fromValues(0.175, -0.1, -1.0, 1.0), 0.1, vec4.fromValues(0.2, 0.9, 0.5, 1.0), 0.0));
  scene.addObject(createSphere(vec4.fromValues(0.3, -0.1, -1.2, 1.0), 0.1, vec4.fromValues(0.2, 0.9, 0.5, 1.0), 0.0));

  // stripes
  scene.addObject(createSphere(vec4.fromValues(-0.125, -0.09, -0.815, 1.0), 0.08, vec4.fromValues(0.2, 0.6, 0.5, 1.0), 0.0));
  scene.addObject(createSphere(vec4.fromValues(-0.01, -0.1, -0.87, 1.0), 0.08, vec4.fromValues(0.2, 0.6, 0.5, 1.0), 0.0));
  scene.addObject(createSphere(vec4.fromValues(0.110, -0.1, -0.95, 1.0), 0.08, vec4.fromValues(0.2, 0.6, 0.5, 1.0), 0.0));
  scene.addObject(createSphere(vec4.fromValues(0.250, -0.1, -1.1, 1.0), 0.08, vec4.fromValues(0.2, 0.6, 0.5, 1.0), 0.0));

  // eyes
  scene.addObject(createSphere(vec4.fromValues(-0.2, -0.0, -0.7, 1.0), 0.03, vec4.fromValues(1.0, 1.0, 1.0, 1.0), 1.0));
  scene.addObject(createSphere(vec4.fromValues(-0.29, -0.0, -0.73, 1.0), 0.03, vec4.fromValues(1.0, 1.0, 1.0, 1.0), 1.0));

  // nose
  scene.addObject(createSphere(vec4.fromValues(-0.26, -0.06, -0.68, 1.0), 0.03, vec4.fromValues(0.7, 0.1, 0.0, 1.0), 0.0));

  // rock
  scene.addObject(createSphere(vec4.fromValues(0.1, -0.21, -0.45, 1.0),
                            0.1,
                            vec4.fromValues(0.3, 0.3, 0.3, 1.0)));

  scene.addObject(createSphere(vec4.fromValues(0.05, -0.21, -0.4, 1.0),
                            0.06,
                            vec4.fromValues(0.3, 0.3, 0.3, 1.0)));
}

// add spheres for a reflected caterpillar
let addReflectedCaterpillar = function(scene) {
  scene.addObject(createSphere(vec4.fromValues(-0.225, -0.07, 0.28, 1.0), 0.12, vec4.fromValues(0.2, 0.9, 0.5, 1.0), 0.0));
  scene.addObject(createSphere(vec4.fromValues(-0.075, -0.1, 0.34, 1.0), 0.1, vec4.fromValues(0.2, 0.9, 0.5, 1.0), 0.0));
  scene.addObject(createSphere(vec4.fromValues(0.05, -0.1, 0.4, 1.0), 0.1, vec4.fromValues(0.2, 0.9, 0.5, 1.0), 0.0));
  scene.addObject(createSphere(vec4.fromValues(0.175, -0.1, 0.5, 1.0), 0.1, vec4.fromValues(0.2, 0.9, 0.5, 1.0), 0.0));
  scene.addObject(createSphere(vec4.fromValues(0.3, -0.1, 0.7, 1.0), 0.1, vec4.fromValues(0.2, 0.9, 0.5, 1.0), 0.0));

  // // stripes
  // scene.addObject(createSphere(vec4.fromValues(-0.125, -0.09, 0.315, 1.0), 0.08, vec4.fromValues(0.2, 0.6, 0.5, 1.0), 0.0));
  // scene.addObject(createSphere(vec4.fromValues(-0.01, -0.1, 0.37, 1.0), 0.08, vec4.fromValues(0.2, 0.6, 0.5, 1.0), 0.0));
  // scene.addObject(createSphere(vec4.fromValues(0.110, -0.1, 0.45, 1.0), 0.08, vec4.fromValues(0.2, 0.6, 0.5, 1.0), 0.0));
  // scene.addObject(createSphere(vec4.fromValues(0.250, -0.1, 0.6, 1.0), 0.08, vec4.fromValues(0.2, 0.6, 0.5, 1.0), 0.0));

  // eyes
  scene.addObject(createSphere(vec4.fromValues(-0.2, -0.0, 0.2, 1.0), 0.03, vec4.fromValues(1.0, 1.0, 1.0, 1.0), 1.0));
  scene.addObject(createSphere(vec4.fromValues(-0.29, -0.0, 0.23, 1.0), 0.03, vec4.fromValues(1.0, 1.0, 1.0, 1.0), 1.0));

  // nose
  scene.addObject(createSphere(vec4.fromValues(-0.26, -0.06, 0.18, 1.0), 0.03, vec4.fromValues(0.7, 0.1, 0.0, 1.0), 0.0));


  // mirror sphere
  scene.addObject(createSphere(vec4.fromValues(0.25, 0.075, -1.0, 1.0),
                            0.6,
                            vec4.fromValues(0.0, 0.0, 0.1, 1.0),
                            3.0));

  // spheres for depth
  scene.addObject(createSphere(vec4.fromValues(0.55, 0.075, 0.0, 1.0),
                            0.3,
                            vec4.fromValues(0.2, 0.9, 0.7, 1.0),
                            1.0));

  scene.addObject(createSphere(vec4.fromValues(-0.3, 0.4, -1.8, 1.0),
                            0.5,
                            vec4.fromValues(0.8, 0.6, 1.0, 1.0),
                            1.0));

  scene.addObject(createSphere(vec4.fromValues(-0.1, 1.0, -6.7, 1.0),
                            4.1,
                            vec4.fromValues(0.6, 0.0, 1.0, 1.0),
                            1.0));
}

// add 5 floating marbles
let addMarbles = function(scene) {

  scene.addObject(createSphere(vec4.fromValues(0.25, 0.075, -0.43, 1.0),
                            0.1,
                            vec4.fromValues(0.2, 0.9, 0.7, 1.0),
                            1.0));

  scene.addObject(createSphere(vec4.fromValues(0.15, 0.2, -0.7, 1.0),
                            0.1,
                            vec4.fromValues(0.3, 0.7, 1.0, 1.0),
                            1.0));

  scene.addObject(createSphere(vec4.fromValues(-0.02, 0.3, -1.1, 1.0),
                            0.1,
                            vec4.fromValues(0.8, 0.3, 1.0, 1.0),
                            1.0));

  scene.addObject(createSphere(vec4.fromValues(-0.3, 0.4, -1.8, 1.0),
                            0.1,
                            vec4.fromValues(0.8, 0.6, 1.0, 1.0),
                            1.0));

  scene.addObject(createSphere(vec4.fromValues(-0.7, 0.4, -2.7, 1.0),
                            0.1,
                            vec4.fromValues(0.6, 0.7, 1.0, 1.0),
                            1.0));


}

// add 2 floating marbles
let addTwoMarbles = function(scene) {

  scene.addObject(createSphere(vec4.fromValues(0.25, 0.075, -0.43, 1.0),
                            0.1,
                            vec4.fromValues(0.2, 0.9, 0.7, 1.0),
                            1.0));

  scene.addObject(createSphere(vec4.fromValues(0.15, 0.2, -0.7, 1.0),
                            0.1,
                            vec4.fromValues(0.3, 0.7, 1.0, 1.0),
                            1.0));


}

// add 3 triangles
let addTriangles = function(scene) {

  scene.addObject(createTriangle(vec4.fromValues(0,0.1,-0.6,1),
    vec4.fromValues(0.1,-0.1,-0.8,1),
    vec4.fromValues(-0.1,-0.1,-0.8,1),
    vec4.fromValues(0.2, 0.4, 1.0, 1.0),
    2.0));


  scene.addObject(createTriangle(vec4.fromValues(-0.2,0.1,-0.5,1),
    vec4.fromValues(-0.24,-0.1,-0.4,1),
    vec4.fromValues(-0.15,-0.1,-0.6,1),
    vec4.fromValues(1.0, 0.3, 0.4, 1.0),
    2.0));


  scene.addObject(createTriangle(vec4.fromValues(0.2,0.1,-0.5,1),
    vec4.fromValues(0.15,-0.1,-0.6,1),
    vec4.fromValues(0.24,-0.1,-0.4,1),
    vec4.fromValues(0.0, 1.0, 0.4, 1.0),
    2.0));

}

// add cone of 5 mirrors
let addMirrors = function(scene) {

  let precision = 5;
  for (let i = 0; i < precision; i++) {
    scene.addObject(createTriangle(vec4.fromValues(0,0.1,-1.0,1),
      vec4.fromValues(Math.cos(2*Math.PI*(i/precision)),
                        Math.sin(2*Math.PI*(i/precision)),-0.8,1),
      vec4.fromValues(Math.cos(2*Math.PI*((i+1)/precision)),
                        Math.sin(2*Math.PI*((i+1)/precision)),-0.8,1),
      vec4.fromValues(0.1, 0.4, 0.6, 1.0),
      3.0));
  }
}

// add cone of mirrors (pointed in -y direction)
let addCone = function(scene, precision) {

  if (!precision)
    precision = 400;
  for (let i = 0; i < precision; i++) {
    scene.addObject(createTriangle(vec4.fromValues(0.0,-0.15,-0.6,1),
      vec4.fromValues(Math.cos(2*Math.PI*(i/precision))*0.3,0.2,
                        Math.sin(2*Math.PI*(i/precision))*0.3-0.7,1),
      vec4.fromValues(Math.cos(2*Math.PI*((i+1)/precision))*0.3,0.2,
                        Math.sin(2*Math.PI*((i+1)/precision))*0.3-0.7,1),
      vec4.fromValues(0.3, 0.9, 1.0, 1.0),
      2.0));
  }
}


let render = function(canvas, context) {
  let image = context.createImageData(canvas.width, canvas.height);
  let subpixel = parseInt(document.getElementById('subpixel').value);
  let depth = parseInt(document.getElementById('depth').value);
  let width = image.height;
  let aspect = 1.0;
  if (document.getElementById('wide').checked) {
    width = image.width;
    aspect = 1.33;
  }


  let view = createView(Math.PI/3, aspect, 1.0);
  let scene = createScene(view);
  console.log("view:", view.n, view.l, view.r);

  scene.addObject(createPlane(vec4.fromValues(0.0, -0.2, 0.0, 1.0),
                              vec4.fromValues(0.0, 1.0, 0.0, 0.0),
                              vec4.fromValues(1.0, 1.0, 1.0, 1.0),
                              2.0));


  let sceneLayout = parseInt(document.getElementById('scene').value);
  switch (sceneLayout) {
    case 0:
      addTriangles(scene);
      break;
    case 1:
      addTwoMarbles(scene);
      break;
    case 2:
      addMarbles(scene);
      break;
    case 3:
      addCaterpillar(scene);
      break;
    case 4:
      addReflectedCaterpillar(scene);
      break;
  }

  if (document.getElementById('drawmirrors').checked) {
    addMirrors(scene);
  }

  if (document.getElementById('drawcone').checked) {
    addCone(scene, document.getElementById('cone').value);
  }


  let ray;

  for (let j = 0; j < image.height; j++) {

    // update progress every 5 lines
    if (j % 10 == 0) {
      console.log(Math.round(j/image.height*100), "% rendered");
    }

    for (let i = 0; i < width; i++) {

      // reset total color
      let c = vec3.fromValues(0, 0, 0);

      if (subpixel > 1) { // average several rays going through the pixel
        for (let xct = 0; xct < subpixel; xct++) {
          for (let yct = 0; yct < subpixel; yct++) {
            vec3.add(c, c, scene.findColor(rayFromPixel(i-0.5+xct/subpixel, j-0.5+yct/subpixel, width, image.height, view), 0, Infinity, depth));
          }
        }
      } else { // if there is no subpixel averaging, use middle of the pxiel
        vec3.add(c, c, scene.findColor(rayFromPixel(i, j, width, image.height, view), 0, Infinity, depth));
      }


      vec3.scale(c, c, 1/(subpixel*subpixel));

      setPixel(image, i, j, 255*c[0], 255*c[1], 255*c[2]);
    }
  }
  console.log("rendered");
  context.putImageData(image, 0,0);
}



window.onload = function(){

  let canvas = document.getElementById('canvas'),
    context = canvas.getContext("2d");

  render(canvas, context);
  document.getElementById('renderbutton').onclick = function() {
    render(canvas, context);
  }

}
