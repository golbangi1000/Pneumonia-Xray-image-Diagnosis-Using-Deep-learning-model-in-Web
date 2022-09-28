package com.XrayWeb.project;

import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service 
public class memberServiceImpl implements memberService {
	 @Autowired
	 memberDao MemberDao; 
	@Override
	public String create(Map<String, Object> map) { 
		int affectRowCount = this.MemberDao.insert(map);
	    
		if (affectRowCount ==  1) {
	        return map.get("member_no").toString(); 
	    }
	    return null;
	}
	@Override 
	public Map<String, Object> memberlogin(Map<String, Object> map){ 

	    return this.MemberDao.login(map); 
	}
	
}
