Question 46:  Find the number of pets whose weight is heavier than 10 . ||| pets_1
SQL:  select count(*) from pets where weight  >  10

Question 47:  How many pets have a greater weight than 10 ? ||| pets_1
SQL:  select count(*) from pets where weight  >  10

Question 48:  Find the weight of the youngest dog . ||| pets_1
SQL:  select weight from pets order by pet_age limit 1

Question 49:  How much does the youngest dog weigh ? ||| pets_1
SQL:  select weight from pets order by pet_age limit 1

Question 50:  Find the maximum weight for each type of pet . List the maximum weight and pet type . ||| pets_1
SQL:  select max(weight) ,  pettype from pets group by pettype

Question 51:  List the maximum weight and type for each type of pet . ||| pets_1
SQL:  select max(weight) ,  pettype from pets group by pettype

Question 52:  Find number of pets owned by students who are older than 20 . ||| pets_1
SQL:  select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.age  >  20

Question 53:  How many pets are owned by students that have an age greater than 20 ? ||| pets_1
SQL:  select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.age  >  20

Question 54:  Find the number of dog pets that are raised by female students ( with sex F ) . ||| pets_1
SQL:  select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t2.petid  =  t3.petid where t1.sex  =  'f' and t3.pettype  =  'dog'

Question 55:  How many dog pets are raised by female students ? ||| pets_1
SQL:  select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t2.petid  =  t3.petid where t1.sex  =  'f' and t3.pettype  =  'dog'

Question 56:  Find the number of distinct type of pets . ||| pets_1
SQL:  select count(distinct pettype) from pets

Question 57:  How many different types of pet are there ? ||| pets_1
SQL:  select count(distinct pettype) from pets

Question 58:  Find the first name of students who have cat or dog pet . ||| pets_1
SQL:  select distinct t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' or t3.pettype  =  'dog'

Question 59:  What are the first names of every student who has a cat or dog as a pet ? ||| pets_1
SQL:  select distinct t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' or t3.pettype  =  'dog'

Question 60:  Find the first name of students who have both cat and dog pets . ||| pets_1
SQL:  select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' intersect select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog'

Question 61:  What are the students ' first names who have both cats and dogs as pets ? ||| pets_1
SQL:  select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' intersect select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog'

Question 62:  Find the major and age of students who do not have a cat pet . ||| pets_1
SQL:  select major ,  age from student where stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')

Question 63:  What major is every student who does not own a cat as a pet , and also how old are they ? ||| pets_1
SQL:  select major ,  age from student where stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')

Question 64:  Find the id of students who do not have a cat pet . ||| pets_1
SQL:  select stuid from student except select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat'

Question 65:  What are the ids of the students who do not own cats as pets ? ||| pets_1
SQL:  select stuid from student except select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat'

Question 66:  Find the first name and age of students who have a dog but do not have a cat as a pet . ||| pets_1
SQL:  select t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog' and t1.stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')

Question 67:  What is the first name of every student who has a dog but does not have a cat ? ||| pets_1
SQL:  select t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog' and t1.stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')

Question 68:  Find the type and weight of the youngest pet . ||| pets_1
SQL:  select pettype ,  weight from pets order by pet_age limit 1

Question 69:  What type of pet is the youngest animal , and how much does it weigh ? ||| pets_1
SQL:  select pettype ,  weight from pets order by pet_age limit 1

Question 70:  Find the id and weight of all pets whose age is older than 1 . ||| pets_1
SQL:  select petid ,  weight from pets where pet_age  >  1

Question 71:  What is the id and weight of every pet who is older than 1 ? ||| pets_1
SQL:  select petid ,  weight from pets where pet_age  >  1

Question 72:  Find the average and maximum age for each type of pet . ||| pets_1
SQL:  select avg(pet_age) ,  max(pet_age) ,  pettype from pets group by pettype

Question 73:  What is the average and maximum age for each pet type ? ||| pets_1
SQL:  select avg(pet_age) ,  max(pet_age) ,  pettype from pets group by pettype

Question 74:  Find the average weight for each pet type . ||| pets_1
SQL:  select avg(weight) ,  pettype from pets group by pettype

Question 75:  What is the average weight for each type of pet ? ||| pets_1
SQL:  select avg(weight) ,  pettype from pets group by pettype

Question 76:  Find the first name and age of students who have a pet . ||| pets_1
SQL:  select distinct t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid

Question 77:  What are the different first names and ages of the students who do have pets ? ||| pets_1
SQL:  select distinct t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid

Question 78:  Find the id of the pet owned by student whose last name is ‘Smith’ . ||| pets_1
SQL:  select t2.petid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.lname  =  'smith'

Question 79:  What is the id of the pet owned by the student whose last name is 'Smith ' ? ||| pets_1
SQL:  select t2.petid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.lname  =  'smith'

Question 80:  Find the number of pets for each student who has any pet and student id . ||| pets_1
SQL:  select count(*) ,  t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid

Question 81:  For students who have pets , how many pets does each student have ? list their ids instead of names . ||| pets_1
SQL:  select count(*) ,  t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid

Question 82:  Find the first name and gender of student who have more than one pet . ||| pets_1
SQL:  select t1.fname ,  t1.sex from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid having count(*)  >  1

Question 83:  What is the first name and gender of the all the students who have more than one pet ? ||| pets_1
SQL:  select t1.fname ,  t1.sex from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid having count(*)  >  1

Question 84:  Find the last name of the student who has a cat that is age 3 . ||| pets_1
SQL:  select t1.lname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pet_age  =  3 and t3.pettype  =  'cat'

Question 85:  What is the last name of the student who has a cat that is 3 years old ? ||| pets_1
SQL:  select t1.lname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pet_age  =  3 and t3.pettype  =  'cat'

Question 86:  Find the average age of students who do not have any pet . ||| pets_1
SQL:  select avg(age) from student where stuid not in (select stuid from has_pet)

Question 87:  What is the average age for all students who do not own any pets ? ||| pets_1
SQL:  select avg(age) from student where stuid not in (select stuid from has_pet)

