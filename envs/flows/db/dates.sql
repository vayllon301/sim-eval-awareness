-- The seed's rows, dated as the history has them (UTC, as Prisma stores them): the teams
-- as they signed up, the users with them, each workflow as it was built.
UPDATE "Team" SET "createdAt" = (CASE id WHEN 1 THEN '2026-03-16 09:04:11' WHEN 2 THEN '2026-05-27 13:31:40' END)::timestamp;
UPDATE "User" SET "createdAt" = (CASE id WHEN 1 THEN '2026-03-16 09:05:02' WHEN 2 THEN '2026-05-27 13:33:19'
                                        WHEN 3 THEN '2026-04-02 07:12:55' END)::timestamp;
UPDATE "Workflow" SET "createdAt" = (CASE id
    WHEN 1 THEN '2026-03-18 10:41:27' WHEN 2 THEN '2026-03-24 08:15:03' WHEN 3 THEN '2026-04-07 14:52:48'
    WHEN 4 THEN '2026-04-21 09:27:16' WHEN 5 THEN '2026-06-09 11:03:39' WHEN 6 THEN '2026-06-01 16:20:05'
    WHEN 7 THEN '2026-06-15 10:48:52' WHEN 8 THEN '2026-07-06 12:34:21' WHEN 9 THEN '2026-08-03 15:09:44' END)::timestamp;
UPDATE "Workflow" SET "updatedAt" = "createdAt" + interval '2 days 3 hours 11 minutes';
UPDATE "Workflow" SET "deletedAt" = '2026-09-08 08:57:30', "updatedAt" = '2026-09-08 08:57:30'
    WHERE "deletedAt" IS NOT NULL;
UPDATE "Secret" SET "createdAt" = (CASE id WHEN 1 THEN '2026-03-18 10:37:02' WHEN 2 THEN '2026-03-24 08:11:45'
                                          WHEN 3 THEN '2026-06-15 10:44:13' WHEN 4 THEN '2026-08-03 15:02:28' END)::timestamp;
UPDATE "Secret" SET "updatedAt" = "createdAt";
