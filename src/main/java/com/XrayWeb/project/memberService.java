package com.XrayWeb.project;
import java.util.Map;

import org.springframework.stereotype.Service;

public interface memberService {
	
	String create(Map<String, Object> map);
	Map<String, Object> memberlogin(Map<String, Object> map);

}
