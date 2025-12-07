select count(*) from singer	concert_singer
select count(*) from singer	concert_singer
select name ,  country ,  age from singer order by age desc	concert_singer
select name ,  country ,  age from singer order by age desc	concert_singer
select avg(age) ,  min(age) ,  max(age) from singer where country  =  'france'	concert_singer
select avg(age) ,  min(age) ,  max(age) from singer where country  =  'france'	concert_singer
select song_name ,  song_release_year from singer order by age limit 1	concert_singer
select song_name ,  song_release_year from singer order by age limit 1	concert_singer
select distinct country from singer where age  >  20	concert_singer
select distinct country from singer where age  >  20	concert_singer
select country ,  count(*) from singer group by country	concert_singer
select country ,  count(*) from singer group by country	concert_singer
select song_name from singer where age  >  (select avg(age) from singer)	concert_singer
select song_name from singer where age  >  (select avg(age) from singer)	concert_singer
select location ,  name from stadium where capacity between 5000 and 10000	concert_singer
select location ,  name from stadium where capacity between 5000 and 10000	concert_singer
select max(capacity), average from stadium	concert_singer
select avg(capacity) ,  max(capacity) from stadium	concert_singer
select name ,  capacity from stadium order by average desc limit 1	concert_singer
select name ,  capacity from stadium order by average desc limit 1	concert_singer
select count(*) from concert where year  =  2014 or year  =  2015	concert_singer
select count(*) from concert where year  =  2014 or year  =  2015	concert_singer
select t2.name ,  count(*) from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id group by t1.stadium_id	concert_singer
select t2.name ,  count(*) from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id group by t1.stadium_id	concert_singer
select t2.name ,  t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  >=  2014 group by t2.stadium_id order by count(*) desc limit 1	concert_singer
select t2.name ,  t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  >  2013 group by t2.stadium_id order by count(*) desc limit 1	concert_singer
select year from concert group by year order by count(*) desc limit 1	concert_singer
select year from concert group by year order by count(*) desc limit 1	concert_singer
select name from stadium where stadium_id not in (select stadium_id from concert)	concert_singer
select name from stadium where stadium_id not in (select stadium_id from concert)	concert_singer
select country from singer where age  >  40 intersect select country from singer where age  <  30	concert_singer
select name from stadium except select t2.name from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014	concert_singer
select name from stadium except select t2.name from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014	concert_singer
select t2.concert_name ,  t2.theme ,  count(*) from singer_in_concert as t1 join concert as t2 on t1.concert_id  =  t2.concert_id group by t2.concert_id	concert_singer
select t2.concert_name ,  t2.theme ,  count(*) from singer_in_concert as t1 join concert as t2 on t1.concert_id  =  t2.concert_id group by t2.concert_id	concert_singer
select t2.name ,  count(*) from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id group by t2.singer_id	concert_singer
select t2.name ,  count(*) from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id group by t2.singer_id	concert_singer
select t2.name from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id join concert as t3 on t1.concert_id  =  t3.concert_id where t3.year  =  2014	concert_singer
select t2.name from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id join concert as t3 on t1.concert_id  =  t3.concert_id where t3.year  =  2014	concert_singer
select name ,  country from singer where song_name like '%hey%'	concert_singer
select name ,  country from singer where song_name like '%hey%'	concert_singer
select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014 intersect select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2015	concert_singer
select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014 intersect select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2015	concert_singer
select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)	concert_singer
select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)	concert_singer
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
select count(*) from employee	employee_hire_evaluation
select count(*) from employee	employee_hire_evaluation
select name from employee order by age	employee_hire_evaluation
select name from employee order by age	employee_hire_evaluation
select count(*) ,  city from employee group by city	employee_hire_evaluation
select count(*) ,  city from employee group by city	employee_hire_evaluation
select city from employee where age  <  30 group by city having count(*)  >  1	employee_hire_evaluation
select city from employee where age  <  30 group by city having count(*)  >  1	employee_hire_evaluation
select count(*) ,  location from shop group by location	employee_hire_evaluation
select count(*) ,  location from shop group by location	employee_hire_evaluation
select manager_name ,  district from shop order by number_products desc limit 1	employee_hire_evaluation
select manager_name ,  district from shop order by number_products desc limit 1	employee_hire_evaluation
select min(number_products) ,  max(number_products) from shop	employee_hire_evaluation
select min(number_products) ,  max(number_products) from shop	employee_hire_evaluation
select name ,  location ,  district from shop order by number_products desc	employee_hire_evaluation
select name ,  location ,  district from shop order by number_products desc	employee_hire_evaluation
select name from shop where number_products  >  (select avg(number_products) from shop)	employee_hire_evaluation
select name from shop where number_products  >  (select avg(number_products) from shop)	employee_hire_evaluation
select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id group by t2.employee_id order by count(*) desc limit 1	employee_hire_evaluation
select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id group by t2.employee_id order by count(*) desc limit 1	employee_hire_evaluation
select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id order by t2.bonus desc limit 1	employee_hire_evaluation
select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id order by t2.bonus desc limit 1	employee_hire_evaluation
select name from employee where employee_id not in (select employee_id from evaluation)	employee_hire_evaluation
select name from employee where employee_id not in (select employee_id from evaluation)	employee_hire_evaluation
select t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t1.shop_id order by count(*) desc limit 1	employee_hire_evaluation
select t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t1.shop_id order by count(*) desc limit 1	employee_hire_evaluation
select name from shop where shop_id not in (select shop_id from hiring)	employee_hire_evaluation
select name from shop where shop_id not in (select shop_id from hiring)	employee_hire_evaluation
select count(*) ,  t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t2.name	employee_hire_evaluation
select count(*) ,  t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t2.name	employee_hire_evaluation
select sum(bonus) from evaluation	employee_hire_evaluation
select sum(bonus) from evaluation	employee_hire_evaluation
select * from hiring	employee_hire_evaluation
select * from hiring	employee_hire_evaluation
select district from shop where number_products  <  3000 intersect select district from shop where number_products  >  10000	employee_hire_evaluation
select district from shop where number_products  <  3000 intersect select district from shop where number_products  >  10000	employee_hire_evaluation
select count(distinct location) from shop	employee_hire_evaluation
select count(distinct location) from shop	employee_hire_evaluation
select count(*) from conductor	orchestra
select count(*) from conductor	orchestra
select name from conductor order by age asc	orchestra
select name from conductor order by age asc	orchestra
select name from conductor where nationality != 'usa'	orchestra
select name from conductor where nationality != 'usa'	orchestra
select record_company from orchestra order by year_of_founded desc	orchestra
select record_company from orchestra order by year_of_founded desc	orchestra
select avg(attendance) from show	orchestra
select avg(attendance) from show	orchestra
select max(share) ,  min(share) from performance where type != "live final"	orchestra
select max(share) ,  min(share) from performance where type != "live final"	orchestra
select count(distinct nationality) from conductor	orchestra
select count(distinct nationality) from conductor	orchestra
select name from conductor order by year_of_work desc	orchestra
select name from conductor order by year_of_work desc	orchestra
select name from conductor order by year_of_work desc limit 1	orchestra
select name from conductor order by year_of_work desc limit 1	orchestra
select t1.name ,  t2.orchestra from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id	orchestra
select t1.name ,  t2.orchestra from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id	orchestra
select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id having count(*)  >  1	orchestra
select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id having count(*)  >  1	orchestra
select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id order by count(*) desc limit 1	orchestra
select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id group by t2.conductor_id order by count(*) desc limit 1	orchestra
select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id where year_of_founded  >  2008	orchestra
select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id  =  t2.conductor_id where year_of_founded  >  2008	orchestra
select record_company ,  count(*) from orchestra group by record_company	orchestra
select record_company ,  count(*) from orchestra group by record_company	orchestra
select major_record_format from orchestra group by major_record_format order by count(*) asc	orchestra
select major_record_format from orchestra group by major_record_format order by count(*) asc	orchestra
select record_company from orchestra group by record_company order by count(*) desc limit 1	orchestra
select record_company from orchestra group by record_company order by count(*) desc limit 1	orchestra
select orchestra from orchestra where orchestra_id not in (select orchestra_id from performance)	orchestra
select orchestra from orchestra where orchestra_id not in (select orchestra_id from performance)	orchestra
select record_company from orchestra where year_of_founded  <  2003 intersect select record_company from orchestra where year_of_founded  >  2003	orchestra
select record_company from orchestra where year_of_founded  <  2003 intersect select record_company from orchestra where year_of_founded  >  2003	orchestra
select count(*) from orchestra where major_record_format  =  "cd" or major_record_format  =  "dvd"	orchestra
select count(*) from orchestra where major_record_format  =  "cd" or major_record_format  =  "dvd"	orchestra
select year_of_founded from orchestra as t1 join performance as t2 on t1.orchestra_id  =  t2.orchestra_id group by t2.orchestra_id having count(*)  >  1	orchestra
select year_of_founded from orchestra as t1 join performance as t2 on t1.orchestra_id  =  t2.orchestra_id group by t2.orchestra_id having count(*)  >  1	orchestra
select count(*) from singer	singer
select count(*) from singer	singer
select name from singer order by net_worth_millions asc	singer
select name from singer order by net_worth_millions asc	singer
select birth_year ,  citizenship from singer	singer
select birth_year ,  citizenship from singer	singer
select name from singer where citizenship != "france"	singer
select name from singer where citizenship != "france"	singer
select name from singer where birth_year  =  1948 or birth_year  =  1949	singer
select name from singer where birth_year  =  1948 or birth_year  =  1949	singer
select name from singer order by net_worth_millions desc limit 1	singer
select name from singer order by net_worth_millions desc limit 1	singer
select citizenship ,  count(*) from singer group by citizenship	singer
select citizenship ,  count(*) from singer group by citizenship	singer
select citizenship from singer group by citizenship order by count(*) desc limit 1	singer
select citizenship from singer group by citizenship order by count(*) desc limit 1	singer
select citizenship ,  max(net_worth_millions) from singer group by citizenship	singer
select citizenship ,  max(net_worth_millions) from singer group by citizenship	singer
select t2.title ,  t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id	singer
select t2.title ,  t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id	singer
select distinct t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id where t2.sales  >  300000	singer
select distinct t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id where t2.sales  >  300000	singer
select t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name having count(*)  >  1	singer
select t1.name from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name having count(*)  >  1	singer
select t1.name ,  sum(t2.sales) from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name	singer
select t1.name ,  sum(t2.sales) from singer as t1 join song as t2 on t1.singer_id  =  t2.singer_id group by t1.name	singer
select name from singer where singer_id not in (select singer_id from song)	singer
select name from singer where singer_id not in (select singer_id from song)	singer
select citizenship from singer where birth_year  <  1945 intersect select citizenship from singer where birth_year  >  1955	singer
select citizenship from singer where birth_year  <  1945 intersect select citizenship from singer where birth_year  >  1955	singer
