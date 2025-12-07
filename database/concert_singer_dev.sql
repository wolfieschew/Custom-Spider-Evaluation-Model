Question 1:  How many singers do we have ? ||| concert_singer
SQL:  select count(*) from singer

Question 2:  What is the total number of singers ? ||| concert_singer
SQL:  select count(*) from singer

Question 3:  Show name , country , age for all singers ordered by age from the oldest to the youngest . ||| concert_singer
SQL:  select name ,  country ,  age from singer order by age desc

Question 4:  What are the names , countries , and ages for every singer in descending order of age ? ||| concert_singer
SQL:  select name ,  country ,  age from singer order by age desc

Question 5:  What is the average , minimum , and maximum age of all singers from France ? ||| concert_singer
SQL:  select avg(age) ,  min(age) ,  max(age) from singer where country  =  'france'

Question 6:  What is the average , minimum , and maximum age for all French singers ? ||| concert_singer
SQL:  select avg(age) ,  min(age) ,  max(age) from singer where country  =  'france'

Question 7:  Show the name and the release year of the song by the youngest singer . ||| concert_singer
SQL:  select song_name ,  song_release_year from singer order by age limit 1

Question 8:  What are the names and release years for all the songs of the youngest singer ? ||| concert_singer
SQL:  select song_name ,  song_release_year from singer order by age limit 1

Question 9:  What are all distinct countries where singers above age 20 are from ? ||| concert_singer
SQL:  select distinct country from singer where age  >  20

Question 10:  What are the different countries with singers above age 20 ? ||| concert_singer
SQL:  select distinct country from singer where age  >  20

Question 11:  Show all countries and the number of singers in each country . ||| concert_singer
SQL:  select country ,  count(*) from singer group by country

Question 12:  How many singers are from each country ? ||| concert_singer
SQL:  select country ,  count(*) from singer group by country

Question 13:  List all song names by singers above the average age . ||| concert_singer
SQL:  select song_name from singer where age  >  (select avg(age) from singer)

Question 14:  What are all the song names by singers who are older than average ? ||| concert_singer
SQL:  select song_name from singer where age  >  (select avg(age) from singer)

Question 15:  Show location and name for all stadiums with a capacity between 5000 and 10000 . ||| concert_singer
SQL:  select location ,  name from stadium where capacity between 5000 and 10000

Question 16:  What are the locations and names of all stations with capacity between 5000 and 10000 ? ||| concert_singer
SQL:  select location ,  name from stadium where capacity between 5000 and 10000

Question 17:  What is the maximum capacity and the average of all stadiums ? ||| concert_singer
SQL:  select max(capacity), average from stadium

Question 18:  What is the average and maximum capacities for all stadiums ? ||| concert_singer
SQL:  select avg(capacity) ,  max(capacity) from stadium

Question 19:  What is the name and capacity for the stadium with highest average attendance ? ||| concert_singer
SQL:  select name ,  capacity from stadium order by average desc limit 1

Question 20:  What is the name and capacity for the stadium with the highest average attendance ? ||| concert_singer
SQL:  select name ,  capacity from stadium order by average desc limit 1

Question 21:  How many concerts are there in year 2014 or 2015 ? ||| concert_singer
SQL:  select count(*) from concert where year  =  2014 or year  =  2015

Question 22:  How many concerts occurred in 2014 or 2015 ? ||| concert_singer
SQL:  select count(*) from concert where year  =  2014 or year  =  2015

Question 23:  Show the stadium name and the number of concerts in each stadium . ||| concert_singer
SQL:  select t2.name ,  count(*) from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id group by t1.stadium_id

Question 24:  For each stadium , how many concerts play there ? ||| concert_singer
SQL:  select t2.name ,  count(*) from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id group by t1.stadium_id

Question 25:  Show the stadium name and capacity with most number of concerts in year 2014 or after . ||| concert_singer
SQL:  select t2.name ,  t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  >=  2014 group by t2.stadium_id order by count(*) desc limit 1

Question 26:  What is the name and capacity of the stadium with the most concerts after 2013 ? ||| concert_singer
SQL:  select t2.name ,  t2.capacity from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  >  2013 group by t2.stadium_id order by count(*) desc limit 1

Question 27:  Which year has most number of concerts ? ||| concert_singer
SQL:  select year from concert group by year order by count(*) desc limit 1

Question 28:  What is the year that had the most concerts ? ||| concert_singer
SQL:  select year from concert group by year order by count(*) desc limit 1

Question 29:  Show the stadium names without any concert . ||| concert_singer
SQL:  select name from stadium where stadium_id not in (select stadium_id from concert)

Question 30:  What are the names of the stadiums without any concerts ? ||| concert_singer
SQL:  select name from stadium where stadium_id not in (select stadium_id from concert)

Question 31:  Show countries where a singer above age 40 and a singer below 30 are from . ||| concert_singer
SQL:  select country from singer where age  >  40 intersect select country from singer where age  <  30

Question 32:  Show names for all stadiums except for stadiums having a concert in year 2014 . ||| concert_singer
SQL:  select name from stadium except select t2.name from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014

Question 33:  What are the names of all stadiums that did not have a concert in 2014 ? ||| concert_singer
SQL:  select name from stadium except select t2.name from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014

Question 34:  Show the name and theme for all concerts and the number of singers in each concert . ||| concert_singer
SQL:  select t2.concert_name ,  t2.theme ,  count(*) from singer_in_concert as t1 join concert as t2 on t1.concert_id  =  t2.concert_id group by t2.concert_id

Question 35:  What are the names , themes , and number of singers for every concert ? ||| concert_singer
SQL:  select t2.concert_name ,  t2.theme ,  count(*) from singer_in_concert as t1 join concert as t2 on t1.concert_id  =  t2.concert_id group by t2.concert_id

Question 36:  List singer names and number of concerts for each singer . ||| concert_singer
SQL:  select t2.name ,  count(*) from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id group by t2.singer_id

Question 37:  What are the names of the singers and number of concerts for each person ? ||| concert_singer
SQL:  select t2.name ,  count(*) from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id group by t2.singer_id

Question 38:  List all singer names in concerts in year 2014 . ||| concert_singer
SQL:  select t2.name from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id join concert as t3 on t1.concert_id  =  t3.concert_id where t3.year  =  2014

Question 39:  What are the names of the singers who performed in a concert in 2014 ? ||| concert_singer
SQL:  select t2.name from singer_in_concert as t1 join singer as t2 on t1.singer_id  =  t2.singer_id join concert as t3 on t1.concert_id  =  t3.concert_id where t3.year  =  2014

Question 40:  what is the name and nation of the singer who have a song having 'Hey ' in its name ? ||| concert_singer
SQL:  select name ,  country from singer where song_name like '%hey%'

Question 41:  What is the name and country of origin of every singer who has a song with the word 'Hey ' in its title ? ||| concert_singer
SQL:  select name ,  country from singer where song_name like '%hey%'

Question 42:  Find the name and location of the stadiums which some concerts happened in the years of both 2014 and 2015 . ||| concert_singer
SQL:  select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014 intersect select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2015

Question 43:  What are the names and locations of the stadiums that had concerts that occurred in both 2014 and 2015 ? ||| concert_singer
SQL:  select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2014 intersect select t2.name ,  t2.location from concert as t1 join stadium as t2 on t1.stadium_id  =  t2.stadium_id where t1.year  =  2015

Question 44:  Find the number of concerts happened in the stadium with the highest capacity . ||| concert_singer
SQL:  select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)

Question 45:  What are the number of concerts that occurred in the stadium with the largest capacity ? ||| concert_singer
SQL:  select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)

