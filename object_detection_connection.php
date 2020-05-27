<?php
// Database connection establishment
 $con=mysqli_connect("localhost","root","","object_detection");
 
// Check connection
 if (mysqli_connect_errno($con)) {
 echo "MySQL database connection failed: " . mysqli_connect_error();
 }
 else echo " Connected Successfully to Object_Detection database";
 
 $image= $_POST["image"];
 $image_url = $_POST["image_url"];
 $image_blob_data = $_POST["image_blob_data"];
 
 $addquery = "INSERT INTO images(image,image_url) VALUES('".$image."','".$image_url."');";
 mysqli_query($con,$addquery) or die("4: INSERT query Failed");
?> 