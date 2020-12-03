use [YouTube Data Project]
go


WITH SENTIMENT_DATA AS (
	SELECT * FROM (
		SELECT [videoID],sentimentValue,commentNum
			FROM (
				SELECT [videoID],sentimentValue,commentNum
					FROM
					(
					SELECT [videoID]
							,CASE WHEN [polarity] < 0 THEN -1 
							WHEN [polarity] > 0 THEN 1
								ELSE 0 END as sentimentValue
							,row_number() over (partition by [videoID] order by [polarity]) as commentNum
						FROM [YouTube Data Project].[dbo].[youTubeVideosSentimentAnalysisNum] 
					  ) a
				WHERE (COMMENTNUM >= 20 AND COMMENTNUM <= 50) AND COMMENTNUM >= 20
				GROUP BY [videoID],sentimentValue,commentNum) b
				) c
				PIVOT(
						MAX(sentimentValue)
						FOR commentNum IN ([20],[21],[22],[23],[24],[25],[26],[27],[28],[29],[30],
											[31],[32],[33],[34],[35],[36],[37],[38],[39],[40])
					 ) p
	WHERE [40] IS NOT NULL)

	SELECT p.videoID,p.[20] as Subjectivity20
					,p.[21] as Subjectivity21
					,p.[22] as Subjectivity22
					,p.[23] as Subjectivity23
					,p.[24] as Subjectivity24
					,p.[25] as Subjectivity25
					,p.[26] as Subjectivity26
					,p.[27] as Subjectivity27
					,p.[28] as Subjectivity28
					,p.[29] as Subjectivity29
					,p.[30] as Subjectivity30
					,p.[31] as Subjectivity31
					,p.[32] as Subjectivity32
					,p.[33] as Subjectivity33
					,p.[34] as Subjectivity34
					,p.[35] as Subjectivity35
					,p.[36] as Subjectivity36
					,p.[37] as Subjectivity37
					,p.[38] as Subjectivity38
					,p.[39] as Subjectivity39
					,p.[40] as Subjectivity40
					,s.[20] as Sentiment20
					,s.[21] as Sentiment21
					,s.[22] as Sentiment22
					,s.[23] as Sentiment23
					,s.[24] as Sentiment24
					,s.[25] as Sentiment25
					,s.[26] as Sentiment26
					,s.[27] as Sentiment27
					,s.[28] as Sentiment28
					,s.[29] as Sentiment29
					,s.[30] as Sentiment30
					,s.[31] as Sentiment31
					,s.[32] as Sentiment32
					,s.[33] as Sentiment33
					,s.[34] as Sentiment34
					,s.[35] as Sentiment35
					,s.[36] as Sentiment36
					,s.[37] as Sentiment37
					,s.[38] as Sentiment38
					,s.[39] as Sentiment39
					,s.[40] as Sentiment40 into YouTubeDataPivoted
			FROM (
				SELECT * 
					FROM (
						SELECT a.[videoID],subjectivity,commentNum
							FROM
							(
							SELECT [videoID]
									,[subjectivity]
									,row_number() over (partition by a.[videoID] order by [polarity]) as commentNum
								FROM [YouTube Data Project].[dbo].[youTubeVideosSentimentAnalysisNum] a
								) a
						WHERE (COMMENTNUM >= 20 AND COMMENTNUM <= 50) AND COMMENTNUM >= 20
						GROUP BY a.[videoID],[subjectivity],commentNum) b
						) c
						PIVOT(
								MAX(subjectivity)
								FOR commentNum IN ([20],[21],[22],[23],[24],[25],[26],[27],[28],[29],[30],
													[31],[32],[33],[34],[35],[36],[37],[38],[39],[40])
								) p
			INNER JOIN SENTIMENT_DATA s ON s.videoID = p.videoID
			WHERE s.[40] IS NOT NULL
			ORDER BY 1