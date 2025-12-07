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
