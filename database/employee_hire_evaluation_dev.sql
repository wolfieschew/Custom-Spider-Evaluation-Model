Question 260:  How many employees are there ? ||| employee_hire_evaluation
SQL:  select count(*) from employee

Question 261:  Count the number of employees ||| employee_hire_evaluation
SQL:  select count(*) from employee

Question 262:  Sort employee names by their age in ascending order . ||| employee_hire_evaluation
SQL:  select name from employee order by age

Question 263:  List the names of employees and sort in ascending order of age . ||| employee_hire_evaluation
SQL:  select name from employee order by age

Question 264:  What is the number of employees from each city ? ||| employee_hire_evaluation
SQL:  select count(*) ,  city from employee group by city

Question 265:  Count the number of employees for each city . ||| employee_hire_evaluation
SQL:  select count(*) ,  city from employee group by city

Question 266:  Which cities do more than one employee under age 30 come from ? ||| employee_hire_evaluation
SQL:  select city from employee where age  <  30 group by city having count(*)  >  1

Question 267:  Find the cities that have more than one employee under age 30 . ||| employee_hire_evaluation
SQL:  select city from employee where age  <  30 group by city having count(*)  >  1

Question 268:  Find the number of shops in each location . ||| employee_hire_evaluation
SQL:  select count(*) ,  location from shop group by location

Question 269:  How many shops are there in each location ? ||| employee_hire_evaluation
SQL:  select count(*) ,  location from shop group by location

Question 270:  Find the manager name and district of the shop whose number of products is the largest . ||| employee_hire_evaluation
SQL:  select manager_name ,  district from shop order by number_products desc limit 1

Question 271:  What are the manager name and district of the shop that sells the largest number of products ? ||| employee_hire_evaluation
SQL:  select manager_name ,  district from shop order by number_products desc limit 1

Question 272:  find the minimum and maximum number of products of all stores . ||| employee_hire_evaluation
SQL:  select min(number_products) ,  max(number_products) from shop

Question 273:  What are the minimum and maximum number of products across all the shops ? ||| employee_hire_evaluation
SQL:  select min(number_products) ,  max(number_products) from shop

Question 274:  Return the name , location and district of all shops in descending order of number of products . ||| employee_hire_evaluation
SQL:  select name ,  location ,  district from shop order by number_products desc

Question 275:  Sort all the shops by number products in descending order , and return the name , location and district of each shop . ||| employee_hire_evaluation
SQL:  select name ,  location ,  district from shop order by number_products desc

Question 276:  Find the names of stores whose number products is more than the average number of products . ||| employee_hire_evaluation
SQL:  select name from shop where number_products  >  (select avg(number_products) from shop)

Question 277:  Which shops ' number products is above the average ? Give me the shop names . ||| employee_hire_evaluation
SQL:  select name from shop where number_products  >  (select avg(number_products) from shop)

Question 278:  find the name of employee who was awarded the most times in the evaluation . ||| employee_hire_evaluation
SQL:  select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id group by t2.employee_id order by count(*) desc limit 1

Question 279:  Which employee received the most awards in evaluations ? Give me the employee name . ||| employee_hire_evaluation
SQL:  select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id group by t2.employee_id order by count(*) desc limit 1

Question 280:  Find the name of the employee who got the highest one time bonus . ||| employee_hire_evaluation
SQL:  select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id order by t2.bonus desc limit 1

Question 281:  Which employee received the biggest bonus ? Give me the employee name . ||| employee_hire_evaluation
SQL:  select t1.name from employee as t1 join evaluation as t2 on t1.employee_id  =  t2.employee_id order by t2.bonus desc limit 1

Question 282:  Find the names of employees who never won any award in the evaluation . ||| employee_hire_evaluation
SQL:  select name from employee where employee_id not in (select employee_id from evaluation)

Question 283:  What are the names of the employees who never received any evaluation ? ||| employee_hire_evaluation
SQL:  select name from employee where employee_id not in (select employee_id from evaluation)

Question 284:  What is the name of the shop that is hiring the largest number of employees ? ||| employee_hire_evaluation
SQL:  select t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t1.shop_id order by count(*) desc limit 1

Question 285:  Which shop has the most employees ? Give me the shop name . ||| employee_hire_evaluation
SQL:  select t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t1.shop_id order by count(*) desc limit 1

Question 286:  Find the name of the shops that do not hire any employee . ||| employee_hire_evaluation
SQL:  select name from shop where shop_id not in (select shop_id from hiring)

Question 287:  Which shops run with no employees ? Find the shop names ||| employee_hire_evaluation
SQL:  select name from shop where shop_id not in (select shop_id from hiring)

Question 288:  Find the number of employees hired in each shop ; show the shop name as well . ||| employee_hire_evaluation
SQL:  select count(*) ,  t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t2.name

Question 289:  For each shop , return the number of employees working there and the name of the shop . ||| employee_hire_evaluation
SQL:  select count(*) ,  t2.name from hiring as t1 join shop as t2 on t1.shop_id  =  t2.shop_id group by t2.name

Question 290:  What is total bonus given in all evaluations ? ||| employee_hire_evaluation
SQL:  select sum(bonus) from evaluation

Question 291:  Find the total amount of bonus given in all the evaluations . ||| employee_hire_evaluation
SQL:  select sum(bonus) from evaluation

Question 292:  Give me all the information about hiring . ||| employee_hire_evaluation
SQL:  select * from hiring

Question 293:  What is all the information about hiring ? ||| employee_hire_evaluation
SQL:  select * from hiring

Question 294:  Which district has both stores with less than 3000 products and stores with more than 10000 products ? ||| employee_hire_evaluation
SQL:  select district from shop where number_products  <  3000 intersect select district from shop where number_products  >  10000

Question 295:  Find the districts in which there are both shops selling less than 3000 products and shops selling more than 10000 products . ||| employee_hire_evaluation
SQL:  select district from shop where number_products  <  3000 intersect select district from shop where number_products  >  10000

Question 296:  How many different store locations are there ? ||| employee_hire_evaluation
SQL:  select count(distinct location) from shop

Question 297:  Count the number of distinct store locations . ||| employee_hire_evaluation
SQL:  select count(distinct location) from shop

