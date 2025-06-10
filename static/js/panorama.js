/*
Copyright (c) 2010, Martin Wengenmayer ( www.cheetah3d.com )
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are 
permitted provided that the following conditions are met:

-Redistributions of source code must retain the above copyright notice, this list of 
conditions and the following disclaimer. 

-Redistributions in binary form must reproduce the above copyright notice, this list 
of conditions and the following disclaimer in the documentation and/or other materials 
provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS 
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER 
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

const 	FPS = 120;
const	DEG2RAD=Math.PI/180.0;

//Canvas to which to draw the panorama
var		pano_canvas=null;
var		depth_canvas=null;
 
//Event state
var		mouseIsDown=false;
var		mouseDownPosLastX=0;
var		mouseDownPosLastY=0;
var		displayInfo=false;
var		highquality=true;

//Camera state
var		cam_heading=90.0;
var		cam_pitch=90.0;
var 	cam_fov=90;

//Load image 
var img_buffer = null;
var img = new Image();
img.onload = imageLoaded;
img.src = '';

//Load image 
var depth_buffer = null;
var depth = new Image();
depth.onload = depthLoaded;
depth.src = '';

// 初始化
function init_pano(){
	//get canvas and set up call backs
	pano_canvas = document.getElementById('canvas');
	pano_canvas.onmousedown = mouseDown;
	pano_canvas.onmousewheel = mouseScroll;
	window.onmousemove = mouseMove;
	window.onmouseup = mouseUp;
	window.onkeydown = keyDown;
	// window.onmousewheel = mouseScroll;
	//setInterval(draw, 1000/FPS);
	depth_canvas = document.getElementById('depth_canvas');
	depth_canvas.onmousedown = mouseDown;
	depth_canvas.onmousewheel = mouseScroll;
    // 初始化下拉框选项
	fetchAndPopulateImages();
}

// 动态请求图像列表并填充下拉框
function fetchAndPopulateImages() {
    const selector = document.getElementById("imageSelector");

    // 请求图像列表
    fetch('/images')
        .then(response => {
            if (!response.ok) {
                throw new Error("Failed to fetch image list");
            }
            return response.json();
        })
        .then(imageList => {
            // 清空下拉框现有选项
            selector.innerHTML = "";

            // 动态添加选项
            imageList.forEach(imageName => {
                const option = document.createElement("option");
                option.value = imageName;
                option.textContent = imageName;
                selector.appendChild(option);
            });

            // 默认加载第一张图像
            if (imageList.length > 0) {
                selector.value = imageList[0];
                updateImageSelection();
            }
        })
        .catch(error => {
            console.error("Error fetching image list:", error);
        });
}

// 当下拉框值改变时更新图像
function updateImageSelection() {
    const selector = document.getElementById("imageSelector");
    const selectedImage = selector.value;

    // 更新图像路径并触发重新加载
    img.src = `./results/pano/${selectedImage}`;
    console.log(`Image switched to: ${img.src}`);

    // 更新深度路径并触发重新加载
    depth.src = `./results/pano_depth/${selectedImage}`;
    console.log(`Depth switched to: ${depth.src}`);

	// 同步更新静态框
	const imageDisplay = document.getElementById("imageDisplay");
    imageDisplay.src = `./results/pano/${selectedImage}`;

	// 同步更新静态框
	const depthDisplay = document.getElementById("depthDisplay");
    depthDisplay.src = `./results/pano_depth/${selectedImage}`;

	// 重置相机
	cam_heading = 90.0;
	cam_pitch = 90.0;
	cam_fov = 90;
}

function imageLoaded(){
	var   buffer = document.createElement("canvas");
	var   buffer_ctx = buffer.getContext("2d");
	
	//set buffer size
	buffer.width = img.width;
	buffer.height = img.height;
 	
 	//draw image
	buffer_ctx.drawImage(img, 0, 0);
 	
 	//get pixels
 	var buffer_imgdata = buffer_ctx.getImageData(0, 0, buffer.width, buffer.height);
 	var buffer_pixels = buffer_imgdata.data;
 	
 	//convert imgdata to float image buffer
 	img_buffer = new Array(img.width * img.height * 3);
 	for(var i = 0, j = 0; i < buffer_pixels.length; i += 4, j += 3){
		img_buffer[j] 	= buffer_pixels[i];
		img_buffer[j+1] = buffer_pixels[i+1];
		img_buffer[j+2] = buffer_pixels[i+2];
	}

	// draw image once loaded
	draw();
}

function depthLoaded(){
	var   buffer = document.createElement("canvas");
	var   buffer_ctx = buffer.getContext("2d");
	
	//set buffer size
	buffer.width = depth.width;
	buffer.height = depth.height;
 	
 	//draw image
	buffer_ctx.drawImage(depth, 0, 0);
 	
 	//get pixels
 	var buffer_depthdata = buffer_ctx.getImageData(0, 0, buffer.width, buffer.height);
 	var buffer_pixels = buffer_depthdata.data;
 	
 	//convert imgdata to float image buffer
 	depth_buffer = new Array(depth.width * depth.height * 3);
 	for(var i = 0, j = 0; i < buffer_pixels.length; i += 4, j += 3){
		depth_buffer[j] 	= buffer_pixels[i];
		depth_buffer[j+1] = buffer_pixels[i+1];
		depth_buffer[j+2] = buffer_pixels[i+2];
	}

	// draw image once loaded
	draw();
}

function mouseDown(e){
	mouseIsDown=true;
	mouseDownPosLastX=e.clientX;
	mouseDownPosLastY=e.clientY;
}

function mouseUp(e){
	mouseIsDown=false;
}

function mouseMove(e){
	if(mouseIsDown==true){
		cam_heading-=(e.clientX-mouseDownPosLastX);
		cam_pitch+=0.5*(e.clientY-mouseDownPosLastY);
		cam_pitch=Math.min(180,Math.max(0,cam_pitch));
		mouseDownPosLastX=e.clientX;
		mouseDownPosLastY=e.clientY;	
		draw();
	}
}

function mouseScroll(e){
	cam_fov+=e.wheelDelta/120;
	cam_fov=Math.min(90,Math.max(30,cam_fov));
	draw();
}

function keyDown(e){
	if(e.keyCode==73){	//i==73 Info
		displayInfo = !displayInfo;
		draw();
	}
}

function renderPanorama(canvas){
	if(canvas != null && img_buffer != null){
		var ctx = canvas.getContext("2d");
		var imgdata = ctx.getImageData(0, 0, canvas.width, canvas.height);
		var pixels = imgdata.data;

		var src_width=img.width;
		var src_height=img.height;
		var dest_width=canvas.width;
		var dest_height=canvas.height;
		
		//calculate camera plane
		var theta_fac=src_height/Math.PI;
		var phi_fac=src_width*0.5/Math.PI
		var ratioUp=2.0*Math.tan(cam_fov*DEG2RAD/2.0);
		var ratioRight=ratioUp*1.33;
		var camDirX=Math.sin(cam_pitch*DEG2RAD)*Math.sin(cam_heading*DEG2RAD);
		var camDirY=Math.cos(cam_pitch*DEG2RAD);
		var camDirZ=Math.sin(cam_pitch*DEG2RAD)*Math.cos(cam_heading*DEG2RAD);
		var camUpX=ratioUp*Math.sin((cam_pitch-90.0)*DEG2RAD)*Math.sin(cam_heading*DEG2RAD);
		var camUpY=ratioUp*Math.cos((cam_pitch-90.0)*DEG2RAD);
		var camUpZ=ratioUp*Math.sin((cam_pitch-90.0)*DEG2RAD)*Math.cos(cam_heading*DEG2RAD);
		var camRightX=ratioRight*Math.sin((cam_heading-90.0)*DEG2RAD);
		var camRightY=0.0;
		var camRightZ=ratioRight*Math.cos((cam_heading-90.0)*DEG2RAD);
		var camPlaneOriginX=camDirX + 0.5*camUpX - 0.5*camRightX;
		var camPlaneOriginY=camDirY + 0.5*camUpY - 0.5*camRightY;
		var camPlaneOriginZ=camDirZ + 0.5*camUpZ - 0.5*camRightZ;
		
		//render image
		var	i,j;
		for(i=0;i<dest_height;i++){
			for(j=0;j<dest_width;j++){
				var	fx=j/dest_width;
				var	fy=i/dest_height;
				
				var	rayX=camPlaneOriginX + fx*camRightX - fy*camUpX;
				var	rayY=camPlaneOriginY + fx*camRightY - fy*camUpY;
				var	rayZ=camPlaneOriginZ + fx*camRightZ - fy*camUpZ;
				var	rayNorm=1.0/Math.sqrt(rayX*rayX + rayY*rayY + rayZ*rayZ);
				
				var	theta=Math.acos(rayY*rayNorm);
    			var	phi=Math.atan2(rayZ,rayX) + Math.PI;
    			var	theta_i=Math.floor(theta_fac*theta);
    			var	phi_i=Math.floor(phi_fac*phi);
    			
    			var	dest_offset=4*(i*dest_width+j);
				var	src_offset=3*(theta_i*src_width + phi_i);
				
				pixels[dest_offset]     = img_buffer[src_offset];
				pixels[dest_offset+1]   = img_buffer[src_offset+1];
				pixels[dest_offset+2]   = img_buffer[src_offset+2];
				//pixels[dest_offset+3] = img_buffer[src_offset+3];
			}
		}
 		
 		//upload image data
 		ctx.putImageData(imgdata, 0, 0);
	}
}

function renderPanoramaAndDepth(canvas, depth_canvas){
	if(canvas != null && img_buffer != null && depth_canvas != null){
		var ctx = canvas.getContext("2d");
		var imgdata = ctx.getImageData(0, 0, canvas.width, canvas.height);
		var pixels = imgdata.data;

		var ctx_depth = depth_canvas.getContext("2d");
		var depthdata = ctx_depth.getImageData(0, 0, depth_canvas.width, depth_canvas.height);
		var depth_pixels = depthdata.data;

		var src_width=img.width;
		var src_height=img.height;
		var dest_width=canvas.width;
		var dest_height=canvas.height;
		
		//calculate camera plane
		var theta_fac=src_height/Math.PI;
		var phi_fac=src_width*0.5/Math.PI
		var ratioUp=2.0*Math.tan(cam_fov*DEG2RAD/2.0);
		var ratioRight=ratioUp*1.33;
		var camDirX=Math.sin(cam_pitch*DEG2RAD)*Math.sin(cam_heading*DEG2RAD);
		var camDirY=Math.cos(cam_pitch*DEG2RAD);
		var camDirZ=Math.sin(cam_pitch*DEG2RAD)*Math.cos(cam_heading*DEG2RAD);
		var camUpX=ratioUp*Math.sin((cam_pitch-90.0)*DEG2RAD)*Math.sin(cam_heading*DEG2RAD);
		var camUpY=ratioUp*Math.cos((cam_pitch-90.0)*DEG2RAD);
		var camUpZ=ratioUp*Math.sin((cam_pitch-90.0)*DEG2RAD)*Math.cos(cam_heading*DEG2RAD);
		var camRightX=ratioRight*Math.sin((cam_heading-90.0)*DEG2RAD);
		var camRightY=0.0;
		var camRightZ=ratioRight*Math.cos((cam_heading-90.0)*DEG2RAD);
		var camPlaneOriginX=camDirX + 0.5*camUpX - 0.5*camRightX;
		var camPlaneOriginY=camDirY + 0.5*camUpY - 0.5*camRightY;
		var camPlaneOriginZ=camDirZ + 0.5*camUpZ - 0.5*camRightZ;
		
		//render image
		var	i,j;
		for(i=0;i<dest_height;i++){
			for(j=0;j<dest_width;j++){
				var	fx=j/dest_width;
				var	fy=i/dest_height;
				
				var	rayX=camPlaneOriginX + fx*camRightX - fy*camUpX;
				var	rayY=camPlaneOriginY + fx*camRightY - fy*camUpY;
				var	rayZ=camPlaneOriginZ + fx*camRightZ - fy*camUpZ;
				var	rayNorm=1.0/Math.sqrt(rayX*rayX + rayY*rayY + rayZ*rayZ);
				
				var	theta=Math.acos(rayY*rayNorm);
    			var	phi=Math.atan2(rayZ,rayX) + Math.PI;
    			var	theta_i=Math.floor(theta_fac*theta);
    			var	phi_i=Math.floor(phi_fac*phi);
    			
    			var	dest_offset=4*(i*dest_width+j);
				var	src_offset=3*(theta_i*src_width + phi_i);
				
				pixels[dest_offset]     = img_buffer[src_offset];
				pixels[dest_offset+1]   = img_buffer[src_offset+1];
				pixels[dest_offset+2]   = img_buffer[src_offset+2];
				//pixels[dest_offset+3] = img_buffer[src_offset+3];

				if (depth_buffer != null){
					depth_pixels[dest_offset]     = depth_buffer[src_offset];
					depth_pixels[dest_offset+1]   = depth_buffer[src_offset+1];
					depth_pixels[dest_offset+2]   = depth_buffer[src_offset+2];
					// depth_pixels[dest_offset+3] = depth_buffer[src_offset+3];
				}
			}
		}
 		
 		//upload image data
 		ctx.putImageData(imgdata, 0, 0);
 		ctx_depth.putImageData(depthdata, 0, 0);
	}
}

function drawRoundedRect(ctx,ox,oy,w,h,radius){
	ctx.beginPath();
	ctx.moveTo(ox + radius,oy);
	ctx.lineTo(ox + w - radius,oy);
	ctx.arc(ox +w-radius,oy+ radius, radius,-Math.PI/2,0, false);
	ctx.lineTo(ox + w,oy + h - radius);
	ctx.arc(ox +w-radius,oy + h - radius, radius,0,Math.PI/2, false);
	ctx.lineTo(ox + radius,oy + h);
	ctx.arc(ox + radius,oy + h - radius, radius,Math.PI/2,Math.PI, false);
	ctx.lineTo(ox,oy + radius);
	ctx.arc(ox + radius,oy + radius, radius,Math.PI,3*Math.PI/2, false);
	ctx.fill();	
}

function draw(){
    if(pano_canvas != null && pano_canvas.getContext != null && depth_canvas != null && depth_canvas.getContext != null){
    	var ctx = pano_canvas.getContext("2d");
    	var ctx_depth = depth_canvas.getContext("2d");
    	
    	//clear canvas
    	ctx.fillStyle = "rgba(0, 0, 0, 1)";
    	ctx.fillRect(0, 0, pano_canvas.width, pano_canvas.height);
		
    	//clear canvas
    	ctx_depth.fillStyle = "rgba(0, 0, 0, 1)";
    	ctx_depth.fillRect(0, 0, depth_canvas.width, depth_canvas.height);
		
		//render paromana direct
		var startTime = new Date();
		// renderPanorama(pano_canvas);
		renderPanoramaAndDepth(pano_canvas, depth_canvas);
		var endTime = new Date();
		
		//draw info text
		if(displayInfo==true){	
			ctx.fillStyle = "rgba(255,255,255,0.75)";
			drawRoundedRect(ctx, 20, pano_canvas.height - 60 - 20, 180, 60, 7);
			
			ctx.fillStyle = "rgba(0, 0, 0, 1)";
			ctx.font="10pt helvetica";
			ctx.fillText("Pers size = " + pano_canvas.width + "x" + pano_canvas.height, 30, pano_canvas.height - 60);
			ctx.fillText("Pano size = " + img.width + "x" + img.height, 30, pano_canvas.height - 45);
			ctx.fillText("FPS = " + (1000.0 / (endTime.getTime() - startTime.getTime())).toFixed(1), 30, pano_canvas.height - 30);
		}
	}
}
