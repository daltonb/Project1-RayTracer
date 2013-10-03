// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
// Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#ifndef INTERSECTIONS_H
#define INTERSECTIONS_H

#include "sceneStructs.h"
#include "cudaMat4.h"

#include "glm/glm.hpp"
#include "utilities.h"
#include <thrust/random.h>

//Some forward declarations
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t);
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v);
__host__ __device__ glm::vec3 getSignOfRay(ray r);
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r);
__host__ __device__ float getIntersection(ray r, staticGeom* geoms, int numberOfGeoms, material* materials, intersection& intersect);
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ void boxIntersectionColor(staticGeom box, ray r, glm::vec3& color);
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal);
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed);

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Quick and dirty epsilon check
__host__ __device__ bool epsilonCheck(float a, float b){
    if(fabs(fabs(a)-fabs(b))<EPSILON){
        return true;
    }else{
        return false;
    }
}

//Self explanatory
__host__ __device__ glm::vec3 getPointOnRay(ray r, float t){
  return r.origin + float(t-.0001)*glm::normalize(r.direction);
}

//LOOK: This is a custom function for multiplying cudaMat4 4x4 matrixes with vectors.
//This is a workaround for GLM matrix multiplication not working properly on pre-Fermi NVIDIA GPUs.
//Multiplies a cudaMat4 matrix and a vec4 and returns a vec3 clipped from the vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//Gets 1/direction for a ray
__host__ __device__ glm::vec3 getInverseDirectionOfRay(ray r){
  return glm::vec3(1.0/r.direction.x, 1.0/r.direction.y, 1.0/r.direction.z);
}

//Gets sign of each component of a ray's inverse direction
__host__ __device__ glm::vec3 getSignOfRay(ray r){
  glm::vec3 inv_direction = getInverseDirectionOfRay(r);
  return glm::vec3((int)(inv_direction.x < 0), (int)(inv_direction.y < 0), (int)(inv_direction.z < 0));
}

__host__ __device__ float getIntersection(ray r, staticGeom* geoms, int numberOfGeoms, material* materials, raySegment* rs) {
  int i;
  int intersect_i;
  rs->intersect.t = -1; // initialize intersection
  rs->intersect.incident = r.direction;
  float tmp = -1;
  for (i=0; i<numberOfGeoms; i++) {
	  staticGeom geom = geoms[i];
	  if (geom.type == SPHERE) {
	    tmp = sphereIntersectionTest(geom, r, rs->intersect.point, rs->intersect.normal);
	  } else if (geom.type == CUBE) {
      tmp = boxIntersectionTest(geom, r, rs->intersect.point, rs->intersect.normal);
    }
    if (tmp > 0) { // we have an intersection
      if (rs->intersect.t > 0) { // intersection already detected
        if (tmp < rs->intersect.t) {
          rs->intersect.t = tmp; // this one is closer
          intersect_i = i;
        }
      } else { // no intersection detected yet
        rs->intersect.t = tmp;
        intersect_i = i;
      }
    }
	}
  if (rs->intersect.t > 0) {
    staticGeom geom = geoms[intersect_i];
    rs->intersect.mat = materials[geom.materialid];
    rs->intersect.color = rs->intersect.mat.color;
    if (geom.hasTexture) {
      boxIntersectionColor(geom, r, rs->intersect.color);
    }
    rs->intersect.emittance = rs->intersect.mat.emittance;
  }
  return rs->intersect.t;
}

__host__ __device__ void boxIntersectionColor(staticGeom box, ray r, glm::vec3& color) {
  // standard cube is axis-aligned, origin-centered, unit side length
  float radius = .5;

  // shift camera to "object space" instead of shifting object to "world space"
  glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

  // initialize vars
  ray rt; rt.origin = ro; rt.direction = rd; // transformed ray
  float t; // parametric position on rt

  glm::vec3 p; // intersection point
  glm::vec3 norm; // intersection normal
  bool intersectFlag = false;

  float min_t; // closest intersection
  glm::vec3 min_norm; // norm of closest intersection

  // for the plane x =  0.5, t = ( 0.5 - ro.x)/(rd.x)
  // for the plane x = -0.5, t = (-0.5 - ro.x)/(rd.x)
  // therefore, when rd.x > 0, x = -0.5 will be closer, and vice-versa
  // NOTE: this does not account for being inside or past the box
  int min_face = 0;
  int face = 0;
  if (!epsilonCheck(rt.direction.x, 0.0f)) { // avoid divide by 0 error
	// get appropriate YZ plane intersection
    if (rt.direction.x < 0) {
      t = ( radius - rt.origin.x)/rt.direction.x;
	    face = 1;
    } else {
      t = (-radius - rt.origin.x)/rt.direction.x;
	    face = 2;
    }
    p = getPointOnRay(rt, t);
	// check for YZ face intersection
    if ((t>0) && (p.y >= -radius) && (p.y <= radius) && (p.z >= -radius) && (p.z <= radius)) {
      min_t = t;
      min_face = face;
	    intersectFlag = true;
    }
  }
  if (!intersectFlag && !epsilonCheck(rt.direction.y, 0.0f)) {
	// get appropriate XZ plane intersection
    if (rt.direction.y < 0) {
      t = ( radius - rt.origin.y)/rt.direction.y;
      face = 3;
    } else {
      t = (-radius - rt.origin.y)/rt.direction.y;
	    face = 4;
    }
    p = getPointOnRay(rt, t);
	// check for XZ face intersection
    if ((t>0) && (p.x >= -radius) && (p.x <= radius) && (p.z >= -radius) && (p.z <= radius)) {
	    if (t < min_t || !intersectFlag) {
		    min_t = t;
		    min_face = face;
        if (!intersectFlag) intersectFlag = true;
	    }
    }
  }
  if (!intersectFlag && !epsilonCheck(rt.direction.z, 0.0f)) {
	// get appropriate XY plane intersection
    if (rt.direction.z < 0) {
      t = ( radius - rt.origin.z)/rt.direction.z;
	    face = 5;
    } else {
      t = (-radius - rt.origin.z)/rt.direction.z;
	    face = 6;
    }
    p = getPointOnRay(rt, t);
	// check for XY face intersection
    if ((t>0) && (p.x >= -radius) && (p.x <= radius) && (p.y >= -radius) && (p.y <= radius)) {
	    if (t < min_t || !intersectFlag) {
		    min_t = t;
		    min_face = face;
        if (!intersectFlag) intersectFlag = true;
	    }
    }
  }

  if (min_face == 5) {
    // shift intersection back to "world space"
    glm::vec3 intersectionPoint = getPointOnRay(rt, min_t);
    float vert = intersectionPoint.y;
    float horz = intersectionPoint.x;
    int row = (float)(0.5-vert) * (box.textureRes.y-1);
    int col = (float)(horz-0.5) * (box.textureRes.x-1);
    color = box.texture[row*(int)box.textureRes.x+col];
  }
  return;
}

//DALTON: DONE
//Cube intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float boxIntersectionTest(staticGeom box, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  // standard cube is axis-aligned, origin-centered, unit side length
  float radius = .5;

  // shift camera to "object space" instead of shifting object to "world space"
  glm::vec3 ro = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

  // initialize vars
  ray rt; rt.origin = ro; rt.direction = rd; // transformed ray
  float t; // parametric position on rt

  glm::vec3 p; // intersection point
  glm::vec3 norm; // intersection normal
  bool intersectFlag = false;

  float min_t; // closest intersection
  glm::vec3 min_norm; // norm of closest intersection

  // for the plane x =  0.5, t = ( 0.5 - ro.x)/(rd.x)
  // for the plane x = -0.5, t = (-0.5 - ro.x)/(rd.x)
  // therefore, when rd.x > 0, x = -0.5 will be closer, and vice-versa
  // NOTE: this does not account for being inside or past the box
  if (!epsilonCheck(rt.direction.x, 0.0f)) { // avoid divide by 0 error
	// get appropriate YZ plane intersection
    if (rt.direction.x < 0) {
      t = ( radius - rt.origin.x)/rt.direction.x;
	    norm = glm::vec3(1, 0, 0);
    } else {
      t = (-radius - rt.origin.x)/rt.direction.x;
	    norm = glm::vec3(-1, 0, 0);
    }
    p = getPointOnRay(rt, t);
	// check for YZ face intersection
    if ((t>0) && (p.y >= -radius) && (p.y <= radius) && (p.z >= -radius) && (p.z <= radius)) {
      min_t = t;
	    min_norm = norm;
	    intersectFlag = true;
    }
  }
  if (!intersectFlag && !epsilonCheck(rt.direction.y, 0.0f)) {
	// get appropriate XZ plane intersection
    if (rt.direction.y < 0) {
      t = ( radius - rt.origin.y)/rt.direction.y;
	    norm = glm::vec3(0, 1, 0);
    } else {
      t = (-radius - rt.origin.y)/rt.direction.y;
	    norm = glm::vec3(0, -1, 0);
    }
    p = getPointOnRay(rt, t);
	// check for XZ face intersection
    if ((t>0) && (p.x >= -radius) && (p.x <= radius) && (p.z >= -radius) && (p.z <= radius)) {
	    if (t < min_t || !intersectFlag) {
		    min_t = t;
		    min_norm = norm;
        if (!intersectFlag) intersectFlag = true;
	    }
    }
  }
  if (!intersectFlag && !epsilonCheck(rt.direction.z, 0.0f)) {
	// get appropriate XY plane intersection
    if (rt.direction.z < 0) {
      t = ( radius - rt.origin.z)/rt.direction.z;
	    norm = glm::vec3(0, 0, 1);
    } else {
      t = (-radius - rt.origin.z)/rt.direction.z;
	    norm = glm::vec3(0, 0, -1);
    }
    p = getPointOnRay(rt, t);
	// check for XY face intersection
    if ((t>0) && (p.x >= -radius) && (p.x <= radius) && (p.y >= -radius) && (p.y <= radius)) {
	    if (t < min_t || !intersectFlag) {
		    min_t = t;
		    min_norm = norm;
        if (!intersectFlag) intersectFlag = true;
	    }
    }
  }

  if (!intersectFlag) return -1;

  // shift intersection back to "world space"
  glm::vec3 realIntersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(rt, min_t), 1.0f));

  intersectionPoint = realIntersectionPoint;
  normal = glm::normalize(multiplyMV(box.transform, glm::vec4(min_norm, 0.0f)));
        
  return glm::length(r.origin - realIntersectionPoint);
}

//LOOK: Here's an intersection test example from a sphere. Now you just need to figure out cube and, optionally, triangle.
//Sphere intersection test, return -1 if no intersection, otherwise, distance to intersection
__host__ __device__ float sphereIntersectionTest(staticGeom sphere, ray r, glm::vec3& intersectionPoint, glm::vec3& normal){
  
  float radius = .5;

  glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin,1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction,0.0f)));

  ray rt; rt.origin = ro; rt.direction = rd;
  
  float vDotDirection = glm::dot(rt.origin, rt.direction);
  float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - pow(radius, 2));
  if (radicand < 0){
    return -1;
  }
  
  float squareRoot = sqrt(radicand);
  float firstTerm = -vDotDirection;
  float t1 = firstTerm + squareRoot;
  float t2 = firstTerm - squareRoot;
  
  float t = 0;
  if (t1 < 0 && t2 < 0) {
      return -1;
  } else if (t1 > 0 && t2 > 0) {
      t = min(t1, t2);
  } else {
      t = max(t1, t2);
  }

  glm::vec3 realIntersectionPoint = multiplyMV(sphere.transform, glm::vec4(getPointOnRay(rt, t), 1.0));
  glm::vec3 realOrigin = multiplyMV(sphere.transform, glm::vec4(0,0,0,1));

  intersectionPoint = realIntersectionPoint;
  normal = glm::normalize(realIntersectionPoint - realOrigin);
        
  return glm::length(r.origin - realIntersectionPoint);
}

//returns x,y,z half-dimensions of tightest bounding box
__host__ __device__ glm::vec3 getRadiuses(staticGeom geom){
    glm::vec3 origin = multiplyMV(geom.transform, glm::vec4(0,0,0,1));
    glm::vec3 xmax = multiplyMV(geom.transform, glm::vec4(.5,0,0,1));
    glm::vec3 ymax = multiplyMV(geom.transform, glm::vec4(0,.5,0,1));
    glm::vec3 zmax = multiplyMV(geom.transform, glm::vec4(0,0,.5,1));
    float xradius = glm::distance(origin, xmax);
    float yradius = glm::distance(origin, ymax);
    float zradius = glm::distance(origin, zmax);
    return glm::vec3(xradius, yradius, zradius);
}

//LOOK: Example for generating a random point on an object using thrust.
//Generates a random point on a given cube
__host__ __device__ glm::vec3 getRandomPointOnCube(staticGeom cube, float randomSeed){

    thrust::default_random_engine rng(hash(randomSeed));
    thrust::uniform_real_distribution<float> u01(0,1);
    thrust::uniform_real_distribution<float> u02(-0.5,0.5);

    //get surface areas of sides
    glm::vec3 radii = getRadiuses(cube);
    float side1 = radii.x * radii.y * 4.0f; //x-y face
    float side2 = radii.z * radii.y * 4.0f; //y-z face
    float side3 = radii.x * radii.z* 4.0f; //x-z face
    float totalarea = 2.0f * (side1+side2+side3);
    
    //pick random face, weighted by surface area
    float russianRoulette = (float)u01(rng);
    
    glm::vec3 point = glm::vec3(.5,.5,.5);
    
    if(russianRoulette<(side1/totalarea)){
        //x-y face
        point = glm::vec3((float)u02(rng), (float)u02(rng), .5);
    }else if(russianRoulette<((side1*2)/totalarea)){
        //x-y-back face
        point = glm::vec3((float)u02(rng), (float)u02(rng), -.5);
    }else if(russianRoulette<(((side1*2)+(side2))/totalarea)){
        //y-z face
        point = glm::vec3(.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2))/totalarea)){
        //y-z-back face
        point = glm::vec3(-.5, (float)u02(rng), (float)u02(rng));
    }else if(russianRoulette<(((side1*2)+(side2*2)+(side3))/totalarea)){
        //x-z face
        point = glm::vec3((float)u02(rng), .5, (float)u02(rng));
    }else{
        //x-z-back face
        point = glm::vec3((float)u02(rng), -.5, (float)u02(rng));
    }
    
    glm::vec3 randPoint = multiplyMV(cube.transform, glm::vec4(point,1.0f));

    return randPoint;
       
}

//TODO: IMPLEMENT THIS FUNCTION
//Generates a random point on a given sphere
__host__ __device__ glm::vec3 getRandomPointOnSphere(staticGeom sphere, float randomSeed){

  return glm::vec3(0,0,0);
}

#endif


