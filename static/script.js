let crowdData = [];
let peakCount = 0;
let currentMode = "";
let labels = [];
let chart;
let lastTime =
performance.now();


let video;

let canvas;

let ctx;

async function startCamera(){
    console.log("CAMERA STARTED");
    currentMode = "camera";
    const stream =
    await navigator.mediaDevices.getUserMedia({
        video:true
    });

    video.srcObject = stream;

    video.onloadedmetadata = () => {

        canvas.width = video.videoWidth;

        canvas.height = video.videoHeight;

        detectFrame();
    }
}

async function detectFrame(){
    
    if(currentMode !== "camera"){
    return;
}
    
    console.log("DETECT FRAME RUNNING");
    const tempCanvas =
    document.createElement('canvas');

    tempCanvas.width = video.videoWidth;

    tempCanvas.height = video.videoHeight;

    const tempCtx =
    tempCanvas.getContext('2d');

    tempCtx.drawImage(video,0,0);

    const imageData =
    tempCanvas.toDataURL('image/jpeg');
    console.log("SENDING FRAME");

    const response = await fetch('/detect' ,{
        

        method:'POST',

        headers:{
            'Content-Type':'application/json'
        },

        body:JSON.stringify({
            image:imageData
        })
    });

    const result = await response.json();

    drawBoxes(result.detections);

    updateAnalytics(result.count);

    requestAnimationFrame(detectFrame);
}
async function detectUploadedVideo(){
    if(currentMode !== "video"){
    return;
}

    if(
        video.paused ||
        video.ended
    ){
        return;
    }

    const tempCanvas =
    document.createElement('canvas');

    tempCanvas.width =
    video.videoWidth;

    tempCanvas.height =
    video.videoHeight;

    const tempCtx =
    tempCanvas.getContext('2d');

    tempCtx.drawImage(
        video,
        0,
        0
    );

    const imageData =
    tempCanvas.toDataURL(
        'image/jpeg'
    );

    try{

        const response =
        await fetch('/detect',{

            method:'POST',

            headers:{
                'Content-Type':'application/json'
            },

            body:JSON.stringify({
                image:imageData
            })
        });

        const result =
        await response.json();

        drawBoxes(
            result.detections
        );

        updateAnalytics(
            result.count
        );

    }catch(error){

        console.log(error);
    }

    requestAnimationFrame(
        detectUploadedVideo
    );
}
function drawBoxes(detections){

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );
    let totalArea = 0;

    detections.forEach(box => {

        // HEATMAP

        const centerX =
        (box.x1 + box.x2) / 2;

        const centerY =
        (box.y1 + box.y2) / 2;

        const gradient =
        ctx.createRadialGradient(

            centerX,
            centerY,
            20,

            centerX,
            centerY,
            120
        );

        gradient.addColorStop(
            0,
            "rgba(255,0,0,0.5)"
        );

        gradient.addColorStop(
            1,
            "rgba(255,0,0,0)"
        );

        ctx.fillStyle =
        gradient;

        ctx.beginPath();

        ctx.arc(
            centerX,
            centerY,
            120,
            0,
            Math.PI * 2
        );

        ctx.fill();

        // BOX

        ctx.strokeStyle =
        "#00ffff";

        ctx.shadowColor =
        "#00ffff";

        ctx.shadowBlur = 20;

        ctx.lineWidth = 3;
        const area =

        (box.x2 - box.x1) *

        (box.y2 - box.y1);

        totalArea += area;

        ctx.strokeRect(
            box.x1,
            box.y1,
            box.x2 - box.x1,
            box.y2 - box.y1
        );

        // LABEL

        ctx.font =
        "20px Arial";

        ctx.fillStyle =
        "#00ffff";

        ctx.fillText(
            "Person",
            box.x1,
            box.y1 - 10
        );
   });

updateDensityByArea(
    totalArea,
    detections.length
);

}
function updateDensityByArea(
    totalArea,
    peopleCount
){

    const frameArea =

    canvas.width *

    canvas.height;

    let occupancy =

(totalArea / frameArea) * 100;

// LIMIT MAX VALUE

occupancy = Math.min(
    occupancy,
    100
);

console.log(
    "Occupancy =",
    occupancy
);

    const densityText =
    document.getElementById(
        "density"
    );
    document
.getElementById(
    "occupancy"
)
.innerText =

occupancy.toFixed(1)
+ "%";
    const alertBox =
    document.getElementById(
        "alertBox"
    );

    // SMART AI LOGIC

    if(
    occupancy < 12 &&
    peopleCount < 10
){

    densityText.innerText =
    "Low";

    alertBox.innerText =
    "✅ AREA SAFE";

    alertBox.style.background =
    "#16a34a";
    alertBox.classList.remove(
    "danger"
);
    
}

else if(
    occupancy < 28
){

    densityText.innerText =
    "Medium";

    alertBox.innerText =
    "⚠ CROWD BUILDING";

    alertBox.style.background =
    "#f59e0b";
    alertBox.classList.remove(
    "danger"
);
}

else{

    densityText.innerText =
    "High";

    alertBox.innerText =
    "🚨 DANGEROUS CROWD";

    alertBox.style.background =
    "#dc2626";
    alertBox.classList.add(
    "danger"
);
}
    // SHOW OCCUPANCY %

    document
    .getElementById(
        "occupancy"
    )
    .innerText =

    occupancy.toFixed(1)
    + "%";
}

function updateAnalytics(count){
    console.log("COUNT =", count);
    if(count > peakCount){

    peakCount = count;

    document
    .getElementById(
        "peakCount"
    )
    .innerText = peakCount;
}

    document.getElementById("count")
    .innerText = count;

    let density = "Low";

    if(count > 10){
        density = "Medium";
    }

    if(count > 20){
        density = "High";
    }

    document.getElementById("density")
    .innerText = density;

const alertBox =
document.getElementById(
    "alertBox"
);

if(alertBox){

    if(count <=1){

    alertBox.innerText =
    "✅ AREA CLEAR";

    alertBox.style.background =
    "#16a34a";
    alertBox.classList.remove(
    "danger"
);
}

else if(count <= 3){

    alertBox.innerText =
    "⚠ CROWD FORMING";

    alertBox.style.background =
    "#f59e0b";
    alertBox.classList.remove(
    "danger"
);
}

else{

    alertBox.innerText =
    "🚨 HIGH CROWD DETECTED";

    alertBox.style.background =
    "#dc2626";
    alertBox.classList.add(
    "danger"
);
}
}
labels.push("");

crowdData.push(count);

if(labels.length > 15){

    labels.shift();

    crowdData.shift();
}

if(chart){

    chart.update();
}
}
window.onload = function(){
    document
document
.getElementById(
    "analyticsBtn"
)
.addEventListener(

    "click",

    function(){

        const analytics =

        document.getElementById(
            "analyticsSection"
        );

        // GLOW EFFECT

        analytics.classList.add(
            "active"
        );

        // BUTTON PRESS

        document
        .getElementById(
            "analyticsBtn"
        )
        .style.transform =
        "scale(0.92)";

        // REMOVE EFFECTS

        setTimeout(()=>{

            analytics.classList.remove(
                "active"
            );

        },1500);

        setTimeout(()=>{

            document
            .getElementById(
                "analyticsBtn"
            )
            .style.transform =
            "scale(1)";

        },200);
    }
);
    video =
document.getElementById(
    "video"
);

canvas =
document.getElementById(
    "overlay"
);

ctx =
canvas.getContext("2d");

    // GRAPH

    const chartCtx =
    document.getElementById("crowdChart");

    chart = new Chart(chartCtx, {

        type:'line',

        data:{

            labels:labels,

            datasets:[{

                label:'Crowd Count',

                data:crowdData,

                borderColor:'#00ffff',

                backgroundColor:'rgba(0,255,255,0.1)',

                borderWidth:3,

                tension:0.4,

                fill:true
            }]
        },

        options:{

            responsive:true,

            animation:false,

            scales:{

                y:{
                    beginAtZero:true
                }
            }
        }
    });

    // IMAGE UPLOAD
document
.getElementById("imageUpload")
.addEventListener(

"change",

async function(e){

    console.log("UPLOAD STARTED");

    const file =
    e.target.files[0];

    console.log(file);

    if(!file){

        alert("No file selected");

        return;
    }

    const reader =
    new FileReader();

    reader.onload = async function(){
    currentMode = "image";
    const uploadedImage =
    document.getElementById(
        "uploadedImage"
    );

    // SHOW IMAGE

    uploadedImage.src =
    reader.result;

    uploadedImage.style.display =
    "block";

    video.style.display =
    "none";

    // WAIT UNTIL IMAGE LOADS

    uploadedImage.onload =
    async function(){

        // MATCH CANVAS SIZE

        canvas.width =
        uploadedImage.naturalWidth;

        canvas.height =
        uploadedImage.naturalHeight;
        canvas.style.width =
        uploadedImage.clientWidth + "px";

        canvas.style.height =
        uploadedImage.clientHeight + "px";
        try{

            // SEND IMAGE TO YOLO

            const response =
            await fetch('/detect',{

                method:'POST',

                headers:{
                    'Content-Type':'application/json'
                },

                body:JSON.stringify({
                    image:reader.result
                })
            });

            const result =
            await response.json();

            // DRAW RESULTS

            drawBoxes(
                result.detections
            );

            updateAnalytics(
                result.count
            );

            alert(
                "People Detected: "
                + result.count
            );

        }catch(error){

            console.log(error);

            alert(error);
        }
    }
}

    reader.readAsDataURL(file);

});

document
.getElementById("cameraBtn")
.addEventListener(

"click",

function(){

    // SHOW CAMERA

    video.style.display =
    "block";

    // HIDE IMAGE

    document
    .getElementById("uploadedImage")
    .style.display = "none";

    // START CAMERA

    startCamera();
});
document
.getElementById("videoUpload")
.addEventListener(

"change",

function(e){

    const file =
    e.target.files[0];

    if(!file) return;

    // HIDE IMAGE

    document
    .getElementById("uploadedImage")
    .style.display = "none";

    // SHOW VIDEO

    video.style.display =
    "block";

    // LOAD VIDEO

    const videoURL =
    URL.createObjectURL(file);

    video.srcObject = null;

    video.src = videoURL;
    currentMode = "video";
    video.onloadedmetadata = async () => {

    await video.play();

    canvas.width =
    video.videoWidth;

    canvas.height =
    video.videoHeight;

    canvas.style.width =
    video.clientWidth + "px";

    canvas.style.height =
    video.clientHeight + "px";

    detectUploadedVideo();
}
});
}
