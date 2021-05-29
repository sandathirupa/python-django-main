// function getFilePath() 
// $("#upload-file").change(function(e)
// {
//     alert("This is called");
//     window.alert("This is called");
//     var x = document.getElementById("upload-file").value;
//     document.getElementById("filePath").innerHTML = "This is test";
// })

function getFilePath()
{
    var x = document.getElementById("upload-file").value;
    document.getElementById("filePath").innerHTML = x;

}