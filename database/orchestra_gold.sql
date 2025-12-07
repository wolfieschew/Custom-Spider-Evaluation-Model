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
