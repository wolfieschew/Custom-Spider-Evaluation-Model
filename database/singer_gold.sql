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
