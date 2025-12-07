select count(*) from pets where weight  >  10	pets_1
select count(*) from pets where weight  >  10	pets_1
select weight from pets order by pet_age limit 1	pets_1
select weight from pets order by pet_age limit 1	pets_1
select max(weight) ,  pettype from pets group by pettype	pets_1
select max(weight) ,  pettype from pets group by pettype	pets_1
select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.age  >  20	pets_1
select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.age  >  20	pets_1
select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t2.petid  =  t3.petid where t1.sex  =  'f' and t3.pettype  =  'dog'	pets_1
select count(*) from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t2.petid  =  t3.petid where t1.sex  =  'f' and t3.pettype  =  'dog'	pets_1
select count(distinct pettype) from pets	pets_1
select count(distinct pettype) from pets	pets_1
select distinct t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' or t3.pettype  =  'dog'	pets_1
select distinct t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' or t3.pettype  =  'dog'	pets_1
select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' intersect select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog'	pets_1
select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat' intersect select t1.fname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog'	pets_1
select major ,  age from student where stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')	pets_1
select major ,  age from student where stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')	pets_1
select stuid from student except select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat'	pets_1
select stuid from student except select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat'	pets_1
select t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog' and t1.stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')	pets_1
select t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'dog' and t1.stuid not in (select t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pettype  =  'cat')	pets_1
select pettype ,  weight from pets order by pet_age limit 1	pets_1
select pettype ,  weight from pets order by pet_age limit 1	pets_1
select petid ,  weight from pets where pet_age  >  1	pets_1
select petid ,  weight from pets where pet_age  >  1	pets_1
select avg(pet_age) ,  max(pet_age) ,  pettype from pets group by pettype	pets_1
select avg(pet_age) ,  max(pet_age) ,  pettype from pets group by pettype	pets_1
select avg(weight) ,  pettype from pets group by pettype	pets_1
select avg(weight) ,  pettype from pets group by pettype	pets_1
select distinct t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid	pets_1
select distinct t1.fname ,  t1.age from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid	pets_1
select t2.petid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.lname  =  'smith'	pets_1
select t2.petid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid where t1.lname  =  'smith'	pets_1
select count(*) ,  t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid	pets_1
select count(*) ,  t1.stuid from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid	pets_1
select t1.fname ,  t1.sex from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid having count(*)  >  1	pets_1
select t1.fname ,  t1.sex from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid group by t1.stuid having count(*)  >  1	pets_1
select t1.lname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pet_age  =  3 and t3.pettype  =  'cat'	pets_1
select t1.lname from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid where t3.pet_age  =  3 and t3.pettype  =  'cat'	pets_1
select avg(age) from student where stuid not in (select stuid from has_pet)	pets_1
select avg(age) from student where stuid not in (select stuid from has_pet)	pets_1
