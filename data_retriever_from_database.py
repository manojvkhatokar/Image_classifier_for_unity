import pymysql
#from pymysql.connector import Error


mydb = pymysql.connect(
    host="localhost",
    user="root",
    passwd="",
    database="object_detection")

print("connection successful")


retreival_query="select image_url from image_s where serial_no = (select max(serial_no) from image_s)" 
mycursor = mydb.cursor()
mycursor.execute(retreival_query)
myresult= mycursor.fetchall()

#for i in myresult:
   # print(i)

image_url_tuple = myresult[0]
image_url_string = image_url_tuple[0]
print (image_url_string)

# =============================================================================
# mycursor2=mydb.cursor()
# result_update_query= "insert into image_s values(%s,%s,%s)"
# mycursor2.execute(result_update_query,(3,image_url_string,'tiger lily'))
# print("value inserted")
# 
# 
#     
# mycursor.execute(retreival_query)
# myresult2= mycursor.fetchall()
# #print((myresult2[0])[0])
# print(myresult)
# =============================================================================







