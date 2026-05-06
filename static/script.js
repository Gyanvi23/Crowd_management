let crowdData = [];

let labels = [];
let chart;


const video = document.getElementById("video");

const canvas = document.getElementById("overlay");

const ctx = canvas.getContext("2d");

async function startCamera(){
    console.log("CAMERA STARTED");
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
}

function updateAnalytics(count){

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
    labels.push("");

crowdData.push(count);

if(labels.length > 15){

    labels.shift();

    crowdData.shift();
}

chart.update();
}
window.onload = function(){

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

async function(){

    // SHOW VIDEO

    video.style.display =
    "block";

    // HIDE IMAGE

    document
    .getElementById(
        "uploadedImage"
    )
    .style.display = "none";

    // CLEAR CANVAS

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    // START CAMERA

    await startCamera();
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