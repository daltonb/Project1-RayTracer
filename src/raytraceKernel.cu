// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//DALTON: DONE
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, int iterations, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  ray r;
  float x_offset;
  float y_offset;
  r.origin = eye;
  switch(iterations % 4) {
    case 0:
      x_offset = SQRT_OF_ONE_THIRTY_SECOND;
      y_offset = SQRT_OF_ONE_EIGHTH;
      break;
    case 1:
      x_offset = -SQRT_OF_ONE_THIRTY_SECOND;
      y_offset = -SQRT_OF_ONE_EIGHTH;
      break;
    case 2:
      x_offset = -SQRT_OF_ONE_EIGHTH;
      y_offset = SQRT_OF_ONE_THIRTY_SECOND;
      break;
    case 3:
      x_offset = SQRT_OF_ONE_EIGHTH;
      y_offset = -SQRT_OF_ONE_THIRTY_SECOND;
      break;
  }
  float x_frac = ((x + x_offset) - resolution.x/2 + 0.5)/resolution.x; // X offset from center in pixels as fraction of image half-width
  float y_frac = (resolution.y/2 - (y + y_offset) - 0.5)/resolution.y; // Y offset from center in pixels as fraction of image half-width
  float x_fov = glm::tan(fov.x*PI/180); // projection of FOV.X onto image plane, 1 unit from eye
  float y_fov = glm::tan(fov.y*PI/180); // projection of FOV.Y onto image plane, 1 unit from eye
  glm::vec3 x_component = x_frac * x_fov * glm::normalize(glm::cross(up, view)); // raycast X component (i.e. pointing to the right)
  glm::vec3 y_component = y_frac * y_fov * glm::normalize(up); // raycast Y component (i.e. pointing up)
  glm::vec3 z_component = glm::normalize(view); // raycast Z component (i.e. pointing into scene)
  r.direction = glm::normalize(x_component + y_component + z_component);
  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//DALTON: IN PROGRESS
//Launch camera rays and compute first intersection
__global__ void traceFirstSegment(glm::vec2 resolution, int iterations, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, raySegment* raypool){


  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){
    raySegment* rs = &(raypool[index]);
    // initialize ray state
    rs->active = true;
    rs->color = glm::vec3(1.0f, 1.0f, 1.0f);
    rs->emittance = 0;
    ray camera_ray = raycastFromCameraKernel(resolution, iterations, x, y, cam.position, cam.view, cam.up, cam.fov);
    getIntersection(camera_ray, geoms, numberOfGeoms, materials, rs);
    // process intersection
    if (rs->intersect.t > 0) {
        rs->color *= rs->intersect.color;
        if (rs->intersect.emittance > 0) {
          rs->emittance = rs->intersect.emittance;
          rs->active = false;
        }
    } else {
      rs->active = false;
    }
  }
}

//Launch next ray and compute next intersection
__global__ void traceNextSegment(glm::vec2 resolution, int iterations, int bounce_count, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, raySegment* raypool){


  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){
    raySegment* rs = &(raypool[index]);
    if (rs->active) {
      ray bounce_ray;
      bounce_ray.origin = rs->intersect.point;
      bounce_ray.direction = getNextSegmentDirection(rs->intersect.mat, rs->intersect.incident, rs->intersect.normal, iterations, index, bounce_count);
      getIntersection(bounce_ray, geoms, numberOfGeoms, materials, rs);
      // process intersection
      if (rs->intersect.t > 0) {
          rs->color *= rs->intersect.color;
          if (rs->intersect.emittance > 0) {
            rs->emittance = rs->intersect.emittance;
            rs->active = false;
          }
      } else {
        rs->active = false;
      }
    }
  }
}

//Calculate emittance * absorption
__global__ void updateColors(glm::vec2 resolution, glm::vec3* colors, raySegment* raypool){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  colors[index] += (raypool[index].color * raypool[index].emittance);
}

// DALTON: DONE
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 2; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and send to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  glm::vec3** texturePointers = new glm::vec3*[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.hasTexture = geoms[i].hasTexture;
    if (newStaticGeom.hasTexture) {
      newStaticGeom.textureRes = geoms[i].textureRes;
      texturePointers[i] = NULL;
      cudaMalloc((void**)&(texturePointers[i]), newStaticGeom.textureRes.x*newStaticGeom.textureRes.y*sizeof(glm::vec3));
      cudaMemcpy(texturePointers[i], geoms[i].texture, sizeof(glm::vec3)*newStaticGeom.textureRes.x*newStaticGeom.textureRes.y, cudaMemcpyHostToDevice);
      newStaticGeom.texture = texturePointers[i];
    }
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }

  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy(cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  //package materials and send to GPU
  material* materialList = new material[numberOfMaterials];
  for(int i=0; i<numberOfMaterials; i++) {
	  material newMaterial;
	  newMaterial.color = materials[i].color;
	  newMaterial.specularExponent = materials[i].specularExponent;
	  newMaterial.specularColor = materials[i].specularColor;
	  newMaterial.hasReflective = materials[i].hasReflective;
	  newMaterial.hasRefractive = materials[i].hasRefractive;
	  newMaterial.indexOfRefraction = materials[i].indexOfRefraction;
	  newMaterial.hasScatter = materials[i].hasScatter;
	  newMaterial.absorptionCoefficient = materials[i].absorptionCoefficient;
	  newMaterial.reducedScatterCoefficient = materials[i].reducedScatterCoefficient;
	  newMaterial.emittance = materials[i].emittance;
	  materialList[i] = newMaterial;
  };

  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy(cudamaterials, materialList, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //create ray segment pool
  raySegment* cudapool = NULL;
  cudaMalloc((void**)&cudapool, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(raySegment));

  thrust::device_vector<int> cudapool_active((int)renderCam->resolution.x*(int)renderCam->resolution.y);
  thrust::sequence(cudapool_active.begin(), cudapool_active.end());

  //kernel launches
  traceFirstSegment<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, cudapool);
  cudaDeviceSynchronize();
  for (int i=0; i<traceDepth-1; i++) {
    traceNextSegment<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, iterations, i+1, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, cudapool);
    cudaDeviceSynchronize();
  }
  updateColors<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, cudaimage, cudapool);
  cudaDeviceSynchronize();
  
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  for (int i=0; i<numberOfGeoms; i++) {
    if (geomList[i].hasTexture) {
      cudaFree(texturePointers[i]);
    }
  }
  cudaFree( cudageoms );
  cudaFree( cudamaterials );
  cudaFree( cudapool );
  delete texturePointers;
  delete geomList;
  delete materialList;

  // make certain the kernel has completed
  cudaDeviceSynchronize(); // cudaThreadSynchronize is deprecated

  checkCUDAError("Kernel failed!");
}
