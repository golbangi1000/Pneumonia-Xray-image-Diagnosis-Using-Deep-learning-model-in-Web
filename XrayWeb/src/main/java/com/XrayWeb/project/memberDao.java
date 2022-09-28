package com.XrayWeb.project;
import java.util.Map;
import org.mybatis.spring.SqlSessionTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

 
@Repository 
public class memberDao {
	 @Autowired
	 SqlSessionTemplate sqlSessionTemplate;
	 
	 public int insert(Map<String, Object> map) {
		  return this.sqlSessionTemplate.insert("member.insert", map);
		}
	 
	 public Map<String, Object> login(Map<String, Object> map) {
		    return this.sqlSessionTemplate.selectOne("member.login", map);
		}
	 
}
